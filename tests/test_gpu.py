"""GPU and CUDA verification test for CakAI Studio."""
import torch


def test_cuda_available():
    assert torch.cuda.is_available(), "CUDA is not available"


def test_gpu_device():
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    assert "NVIDIA" in name or "GeForce" in name


def test_tensor_on_gpu():
    t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    result = t * 2
    expected = torch.tensor([2.0, 4.0, 6.0], device="cuda")
    assert torch.equal(result, expected)


def test_gpu_memory():
    total = torch.cuda.get_device_properties(0).total_memory
    assert total > 0, "GPU memory should be > 0"


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
        print(f"Compute capability: {props.major}.{props.minor}")

        t = torch.tensor([1.0, 2.0, 3.0]).cuda()
        print(f"GPU tensor test: {t * 2}")
        print("\nAll GPU checks passed!")
    else:
        print("CUDA not available!")
