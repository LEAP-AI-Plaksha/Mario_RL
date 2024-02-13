import torch
use_mps = torch.backends.mps.is_available()
print(use_mps)