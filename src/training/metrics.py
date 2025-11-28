import numpy as np
from src.training.loss import NegativePearsonCorrelationLoss


METRICS_TITLES = {
    'r value': 'Pearson Correlation',
    'RMSE': 'Root Mean Square Error',
    'MAE': 'Mean Absolute Error',
    'MAPE': 'Mean Absolute Percentage Error'
}


def calculate_test_metrics(y_pred, y):
    metrics = {}
    loss_fn = NegativePearsonCorrelationLoss()
    metrics['r value'] = 1 - loss_fn(y_pred, y)
    y_pred = y_pred.view(-1).detach().numpy()
    y = y[:len(y_pred)].detach().numpy()
    metrics['RMSE'] = np.mean((y_pred - y) ** 2) ** 0.5
    metrics['MAE'] = np.mean(np.abs(y_pred - y))
    metrics['MAPE'] = np.mean(np.abs((y_pred - y) / y))
    return metrics