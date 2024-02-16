import torch
from torch import nn
use_cuda = torch.cuda.is_available()
print(use_cuda)

use_mps = torch.backends.mps.is_available()
print(use_mps)
