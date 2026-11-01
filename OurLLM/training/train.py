"""
Training loop for OurLLM.
Trains the decoder-only Transformer with causal language modeling.
"""
import sys
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
    def __init__(self, file_path: str, tokenizer: OurBPETokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        lines = [line.strip() for line in text.split("\n") if line.strip()]

        for line in lines:
            ids = tokenizer.encode(line, add_special_tokens=True)
            if len(ids) <= max_length:
                self.samples.append(ids)

        print(f"Loaded {len(self.samples)} samples from {file_path}")

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
    train_files = list(data_dir.glob("*.txt"))
    if not train_files:
        print("No training data found. Run prepare_data.py first.")
        return

    dataset = TextDataset(str(train_files[0]), tokenizer, config["data"]["max_length"])

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
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

    model.train()
    global_step = 0
    total_loss = 0
    start_time = time.time()

    print(f"\nStarting training for {max_steps} steps...")
    print(f"Batch size: {config['training']['batch_size']}, "
          f"Grad accum: {grad_accum_steps}, "
          f"Effective batch: {config['training']['batch_size'] * grad_accum_steps}")

    while global_step < max_steps:
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
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
                print(f"Step {global_step}/{max_steps} | Loss: {avg_loss:.4f} | "
                      f"Speed: {steps_per_sec:.1f} steps/s | "
                      f"Time: {elapsed:.0f}s")
                total_loss = 0

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
    print(f"\nTraining complete! Final checkpoint: {final_path}")
    print(f"Total time: {time.time() - start_time:.0f}s")


if __name__ == "__main__":
    train()
