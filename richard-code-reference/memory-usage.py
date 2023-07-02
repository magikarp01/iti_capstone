#%%
import torch

if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    print("Device ID:", device_id)
else:
    print("No CUDA devices available.")

#%%
torch.cuda.memory_allocated()