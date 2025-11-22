import torch
from torch import nn
from .FSAM import FSAM
from .FeatureExtractor import rPPG_FeatureExtractor


class MyFactorizePhysVector(nn.Module):
    def __init__(self, ch_input, neurons, *args, **kwargs):
        super().__init__()
        self.FeatureExtractor = rPPG_FeatureExtractor(ch_input)
        self.FSAM = FSAM(*args, return_W=True, **kwargs)
        temp = [[nn.Linear(x, y), nn.ReLU()] for x, y in zip(neurons, neurons[1:])]
        self.layers = nn.Sequential(sum(temp, [])[:-1])
    
    def forward(self, x):
        x = self.FeatureExtractor(x)
        W = FSAM(x) # Temporal vector
        y_pred = self.layers(W)
        return y_pred