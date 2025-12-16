# test_torch_gpu_old.py
# Simple script to check PyTorch CUDA visibility and run a small GPU op

import torch

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("No CUDA GPU detected by PyTorch.")
        return

    print("CUDA device count:", torch.cuda.device_count())
    print("Current device index:", torch.cuda.current_device())
    print("Current device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    device = torch.device("cuda")
    a = torch.randn(2000, 2000, device=device)
    b = torch.randn(2000, 2000, device=device)
    c = a @ b

    print("Matmul result shape:", c.shape)
    print("PyTorch GPU test completed successfully.")

if __name__ == "__main__":
    main()
