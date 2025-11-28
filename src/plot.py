import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from src.training.metrics import METRICS_TITLES


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


def plot_test_metrics(metrics_df):
    fig, axes = plt.subplots(1, len(metrics_df.columns), figsize=(14, 3))
    cmap = mpl.colormaps['Set2']
    for metric, ax in zip(metrics_df.columns, axes):
        values = list(metrics_df[metric])
        ax.bar(list(metrics_df.index), values, color=cmap.colors)
        ax.set_ylabel(metric)
        ax.set_title(METRICS_TITLES[metric])
    fig.tight_layout()
    fig.savefig(f'figure/metrics.png')
    return fig, axes