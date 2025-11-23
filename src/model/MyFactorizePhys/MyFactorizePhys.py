from torch import nn
from .FSAM import FSAM
from .FeatureExtractor import rPPG_FeatureExtractor


class MyFactorizePhys(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        x = self.FeatureExtractor(x)
        x = FSAM(x) # Temporal vector