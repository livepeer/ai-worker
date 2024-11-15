import torch
import subprocess

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

# Check system CUDA version
try:
    nvcc_output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
    cuda_version = nvcc_output.split("release ")[-1].split(",")[0]
    print(f"System CUDA version: {cuda_version}")
except:
    print("Unable to check system CUDA version")

# Print the current device
print(f"Current device: {torch.cuda.get_device_name(0)}")
