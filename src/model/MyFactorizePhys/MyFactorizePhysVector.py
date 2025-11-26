import torch
from torch import nn
from .FSAM import FSAM
from .FeatureExtractor import FeatureExtractor


"""
This particular model is not part of my reimplementation of the original FactorizePhys algorithm.

It is instead testing my hypothesis that BVP can be inferred solely from the temporal vector (W)
obtained during NMF, effectively bypassing a majority of the FactorizePhys algorithm.

I do not expect this to work; I am writing this to prove I am wrong for having doubted the authors.
"""


class MyFactorizePhysVector(nn.Module):
    def __init__(self, MFP_model, neurons, *args, **kwargs):
        super().__init__()
        self.MFP = MFP_model
        temp = [[nn.Linear(x, y), nn.ReLU()] for x, y in zip(neurons, neurons[1:])]
        self.fc_layers = nn.Sequential(sum(temp, [])[:-1])
    

    def forward(self, x):
        # First part of FactorizePhys
        x = torch.diff(x, dim=2)
        x = self.MFP.norm(x[:, :3])
        x = self.MFP.FeatureExtractor(x)

        # First part of NetworkHead
        x = self.MFP.NetworkHead.conv1(x)
        x = x - x.min() # Move to zero

        # First part of FSAM up to NMF
        Vst = self.MFP.FSAM.ξ_pre(x) # Convolution
        n, τ, κ, α, β = Vst.shape
        assert n == 1 # FactorizePhys has a leading 1
        Vst_2D = Vst.view(τ, κ * α * β) # 4D -> 2D
        W, H = self.MFP.FSAM.ϕ(Vst_2D, 'WH') # NMF
        
        # Final neural network
        return self.fc_layers(W)