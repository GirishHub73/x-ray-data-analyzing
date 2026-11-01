import sys
import torch

print("Python executable:", sys.executable)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this PyCharm Python environment.")

device = torch.device("cuda:0")
a = torch.randn(2048, 2048, device=device)
b = torch.randn(2048, 2048, device=device)

for _ in range(50):
    result = a @ b

torch.cuda.synchronize()

print("GPU:", torch.cuda.get_device_name(0))
print("Result device:", result.device)
print("PyTorch GPU memory allocated (MiB):",
      round(torch.cuda.memory_allocated() / 1024**2, 1))