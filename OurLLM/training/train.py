"""
Training loop for OurLLM.
Trains the decoder-only Transformer with causal language modeling.
Supports cosine LR schedule, train/eval split, and perplexity tracking.
"""
import sys
import math
import time
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from OurLLM.model.transformer import OurLLMModel
from OurLLM.tokenizer.bpe_tokenizer import OurBPETokenizer, PAD_ID, BOS_ID, EOS_ID


class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer: OurBPETokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for line in texts:
            ids = tokenizer.encode(line, add_special_tokens=True)
            if 4 <= len(ids) <= max_length:
                self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        input_ids = ids[:-1]
        target_ids = ids[1:]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


def collate_fn(batch):
    inputs, targets = zip(*batch)
    max_len = max(len(x) for x in inputs)

    padded_inputs = torch.full((len(inputs), max_len), PAD_ID, dtype=torch.long)
    padded_targets = torch.full((len(targets), max_len), -100, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        padded_inputs[i, :len(inp)] = inp
        padded_targets[i, :len(tgt)] = tgt

    return padded_inputs, padded_targets


def load_texts(data_dir: Path) -> list[str]:
    all_texts = []
    for f in sorted(data_dir.glob("*.txt")):
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    all_texts.append(line)
    return all_texts


def get_cosine_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, eval_loader, loss_fn, device, use_amp):
    model.eval()
    total_loss = 0
    total_tokens = 0
    for input_ids, target_ids in eval_loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        else:
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        num_tokens = (target_ids != -100).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, math.exp(avg_loss)


def train():
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "configs" / "ourllm_config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer_path = project_root / "OurLLM" / "tokenizer" / "trained" / "ourllm_tokenizer.json"
    if not tokenizer_path.exists():
        print(f"Tokenizer not found at {tokenizer_path}")
        print("Run train_tokenizer.py first.")
        return

    tokenizer = OurBPETokenizer()
    tokenizer.load(str(tokenizer_path))

    data_dir = project_root / "OurLLM" / "datasets" / "raw"
    all_texts = load_texts(data_dir)
    if not all_texts:
        print("No training data found. Run prepare_data.py first.")
        return

    import random
    random.seed(42)
    random.shuffle(all_texts)

    split_idx = int(len(all_texts) * config["data"]["train_split"])
    train_texts = all_texts[:split_idx]
    eval_texts = all_texts[split_idx:]

    print(f"Total lines: {len(all_texts)}, Train: {len(train_texts)}, Eval: {len(eval_texts)}")

    train_dataset = TextDataset(train_texts, tokenizer, config["data"]["max_length"])
    eval_dataset = TextDataset(eval_texts, tokenizer, config["data"]["max_length"])

    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = OurLLMModel(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=config["model"]["embedding_dim"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        ff_dim=config["model"]["feedforward_dim"],
        max_seq_length=config["model"]["max_seq_length"],
        dropout=config["model"]["dropout"],
        pad_id=PAD_ID,
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    use_amp = config["training"].get("mixed_precision", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    grad_accum_steps = config["training"]["gradient_accumulation_steps"]

    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    max_steps = config["training"]["max_steps"]
    log_interval = config["training"]["log_interval"]
    checkpoint_interval = config["training"]["checkpoint_interval"]
    eval_interval = config["training"]["eval_interval"]
    warmup_steps = config["training"]["warmup_steps"]
    max_lr = config["training"]["learning_rate"]
    min_lr = config["training"].get("min_lr", 1e-5)

    model.train()
    global_step = 0
    total_loss = 0
    best_eval_loss = float("inf")
    start_time = time.time()

    print(f"\nStarting training for {max_steps} steps...")
    print(f"Batch size: {config['training']['batch_size']}, "
          f"Grad accum: {grad_accum_steps}, "
          f"Effective batch: {config['training']['batch_size'] * grad_accum_steps}")
    print(f"LR schedule: cosine, warmup={warmup_steps}, max_lr={max_lr}, min_lr={min_lr}")

    while global_step < max_steps:
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            lr = get_cosine_lr(global_step, warmup_steps, max_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(input_ids)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                    loss = loss / grad_accum_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1
            else:
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                loss = loss / grad_accum_steps
                loss.backward()

                if (batch_idx + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            total_loss += loss.item() * grad_accum_steps

            if global_step > 0 and global_step % log_interval == 0 and (batch_idx + 1) % grad_accum_steps == 0:
                avg_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                ppl = math.exp(min(avg_loss, 20))
                print(f"Step {global_step}/{max_steps} | Loss: {avg_loss:.4f} | PPL: {ppl:.1f} | "
                      f"LR: {lr:.2e} | Speed: {steps_per_sec:.1f} steps/s | Time: {elapsed:.0f}s")
                total_loss = 0

            if global_step > 0 and global_step % eval_interval == 0 and (batch_idx + 1) % grad_accum_steps == 0:
                eval_loss, eval_ppl = evaluate(model, eval_loader, loss_fn, device, use_amp)
                print(f"  [Eval] Loss: {eval_loss:.4f} | Perplexity: {eval_ppl:.1f}")
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_path = checkpoint_dir / "ourllm_best.pt"
                    torch.save({
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "eval_loss": eval_loss,
                        "eval_ppl": eval_ppl,
                    }, str(best_path))
                    print(f"  New best model saved! (loss={eval_loss:.4f})")
                model.train()

            if global_step > 0 and global_step % checkpoint_interval == 0 and (batch_idx + 1) % grad_accum_steps == 0:
                ckpt_path = checkpoint_dir / f"ourllm_step_{global_step}.pt"
                torch.save({
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                }, str(ckpt_path))
                print(f"Checkpoint saved: {ckpt_path}")

            if global_step >= max_steps:
                break

    final_path = checkpoint_dir / "ourllm_final.pt"
    torch.save({
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, str(final_path))

    eval_loss, eval_ppl = evaluate(model, eval_loader, loss_fn, device, use_amp)
    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"Final checkpoint: {final_path}")
    print(f"Best eval loss: {best_eval_loss:.4f}")
    print(f"Final eval loss: {eval_loss:.4f} | Perplexity: {eval_ppl:.1f}")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    train()
