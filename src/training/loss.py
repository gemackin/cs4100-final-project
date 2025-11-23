from torch import nn
import numpy as np


# Loss function for rPPG used in the paper
class NegativePearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        o_center = output - np.mean(output, axis=0)
        t_center = target - np.mean(target, axis=0)
        o_norm = np.linalg.norm(o_center, ord=2)
        t_norm = np.linalg.norm(t_center, ord=2)
        return 1 - o_center.dot(t_center) / (o_norm * t_norm)