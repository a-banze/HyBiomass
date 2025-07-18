import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F


class NormalizeMeanStd(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""

    # Define init function
    def __init__(self, 
                 mean: Tensor, 
                 std: Tensor):
        super().__init__()
        
        self.mean = mean 
        self.std = std

    @torch.no_grad()  
    def forward(self, x):
        
        x_out = (x - self.mean[None, :, None, None].to(x.device)) / self.std[None, :, None, None].to(x.device)

        return x_out
