import numpy as np
from matplotlib import pyplot as plt


MODEL_TITLES = {
    'FP': 'FactorizePhys',
    'MFP': 'MyFactorizePhys',
    'MFPV': 'MyFactorizePhysVector'
}


def plot_losses(losses, color=None, model_name=None):
    fig, ax = plt.subplots()
    xrange = np.arange(len(losses)) + 1
    ax.plot(xrange, losses, c=color)
    ax.scatter(xrange, losses, s=12, c=color)
    ax.set_ylabel('Average Negative Pearson Correlation Loss')
    ax.set_xlabel('Epoch')
    if model_name:
        ax.set_title(f'Training loss for {MODEL_TITLES[model_name]}')
    fig.savefig(f'figure/loss_{model_name}.png')
    return fig, ax