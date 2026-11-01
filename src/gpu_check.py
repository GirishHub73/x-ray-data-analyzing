import sys
import torch

print("Python executable:", sys.executable)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch CUDA build:", torch.version.cuda)
else:
    raise SystemExit("PyTorch cannot see CUDA in this Python environment.")