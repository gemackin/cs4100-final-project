from torch import nn
import numpy as np


# Loss function for rPPG used in the paper
class NegativePearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        output = output - np.mean(output, axis=0)
        target = target - np.mean(target, axis=0)
        output_norm = np.linalg.norm(output, ord=2)
        target_norm = np.linalg.norm(target, ord=2)
        return 1 - output.dot(target) / (output_norm * target_norm)