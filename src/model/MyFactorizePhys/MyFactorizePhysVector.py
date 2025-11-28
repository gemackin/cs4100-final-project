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


class MFP_NMF(nn.Module):
    def __init__(self, MFP_model):
        super().__init__()
        self.MFP = MFP_model

    def forward(self, x):
        # First part of FactorizePhys
        x = torch.diff(x, dim=2)
        x = self.MFP.norm(x[:, :3])
        print('Norm:', x.shape)
        x = self.MFP.FeatureExtractor(x)
        print('FE:', x.shape)

        # First part of NetworkHead
        x = self.MFP.NetworkHead.conv1(x)
        x = x - x.min() # Move to zero

        # First part of FSAM up to NMF
        Vst = self.MFP.NetworkHead.FSAM.ξ_pre(x) # Convolution
        print('V:', Vst.shape)
        n, τ, κ, α, β = Vst.shape
        assert n == 1 # FactorizePhys has a leading 1
        assert τ == 159 or τ == 160 # This has been an issue
        Vst_2D = Vst.view(τ, κ * α * β) # 4D -> 2D
        print('V2D:', Vst_2D.shape)
        W, H = self.MFP.NetworkHead.FSAM.ϕ(Vst_2D, return_val='WH') # NMF
        raise Exception
        return W, H


class MyFactorizePhysVector(nn.Module):
    def __init__(self, neurons):
        super().__init__()
        temp = [[nn.Linear(x, y), nn.ReLU()] for x, y in zip(neurons, neurons[1:])]
        self.fc_layers = nn.Sequential(*sum(temp, [])[:-1])
    
    def forward(self, x):
        return self.fc_layers(x)

    