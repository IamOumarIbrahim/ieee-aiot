import warnings
warnings.filterwarnings("ignore")
import sys
import torch

def main():
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    print(f"PyTorch {torch.__version__} | CUDA Available: {cuda_avail} | Active Device: {device_name}")

if __name__ == "__main__":
    main()
