import torch

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Print the CUDA version
print("CUDA version:", torch.version.cuda)

# Print the device name
print("Device name:", torch.cuda.get_device_name(0))
