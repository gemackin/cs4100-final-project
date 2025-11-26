from torch import nn
import numpy as np
import torch


# Loss function for rPPG used in the paper
class NegativePearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        output = output.view(-1)
        target = target[:len(output)] # FactorizePhys removes the final frame?
        output -= output.mean(dim=0, keepdim=True) # Centering
        target -= target.mean(dim=0, keepdim=True) # Centering
        cosine = nn.CosineSimilarity(dim=0, eps=1e-6)
        return torch.mean(1 - cosine(output, target))