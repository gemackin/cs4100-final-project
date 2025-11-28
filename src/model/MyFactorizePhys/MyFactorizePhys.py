import torch
import torch.nn as nn
from .FSAM import FSAM
from .FeatureExtractor import ConvolutionNorm, FeatureExtractor


class NetworkHead(nn.Module):
    def __init__(self, ch_align, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Sequential(
            ConvolutionNorm(16, 16),
            ConvolutionNorm(16, 16),
            ConvolutionNorm(16, 16),
            nn.Dropout3d(p=dropout),
        )

        self.FSAM = FSAM(16, ch_align)
        self.norm = nn.InstanceNorm3d(16)
        self.bias = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        conv2_kws = {} # dict(dtype=torch.float64)
        self.conv2 = nn.Sequential(
            ConvolutionNorm(16, 12, **conv2_kws),
            ConvolutionNorm(12, 8, **conv2_kws),
            nn.Conv3d(8, 1, (3, 3, 3), (1, 1, 1), padding=(1, 0, 0), bias=False, **conv2_kws),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x - x.min() # Move to zero
        mask, error = self.FSAM(x)
        mask = mask - mask.min() # Move to zero
        x_factorized = self.norm(torch.mul(x + self.bias, mask + self.bias)) # Residual connections
        output = self.conv2(x + x_factorized).view(-1)
        return output



class MyFactorizePhys(nn.Module):
    def __init__(self, ch_input=3, ch_align=8, dropout=0.1):
        super().__init__()
        self.norm = nn.InstanceNorm3d(ch_input)
        self.FeatureExtractor = FeatureExtractor(ch_input, dropout)
        self.NetworkHead = NetworkHead(ch_align, dropout)
 
    def forward(self, x):
        x = torch.diff(x, dim=2)
        x = self.norm(x[:, :3])
        x = self.FeatureExtractor(x)
        return self.NetworkHead(x)