import torch
from torch import nn
import numpy as np
import tqdm # Progress bar
from torch.autograd import Variable
from src.utils import tprint, average_dictionary
from src.training.metrics import calculate_test_metrics
    

class ModelTrainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)


    # @torch.no_grad()
    def _normalize_tensor(self, y):
        return (y - torch.mean(y)) / torch.std(y)


    def _forward_model(self, X, y):
        X = torch.tensor(X, requires_grad=True).to(self.device)
        X = X.view((1, *X.shape)) # Accounting for funky stuff with FactorizePhys
        y = torch.tensor(y).to(self.device)
        if len(y.shape) > 2: y = y[..., 0] # The authors say there is multi-signal data
        y = self._normalize_tensor(y)
        y_pred = self.model(X)
        if isinstance(y_pred, tuple): y_pred = y_pred[0]
        y_pred = self._normalize_tensor(y_pred)
        return y_pred, y


    def _train_batch(self, X, y):
        y_pred, y = self._forward_model(X, y)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()


    def train(self, dataloader, num_epochs):
        self.model.train() # Set model to training mode
        losses = []
        for epoch in range(num_epochs):
            tprint('Train epoch', epoch + 1)
            losses.append([])
            for X, y, sub_id, split_id in dataloader: # Looping through batches
                # tprint(f'Processing split sub-{sub_id}_split-{split_id}...')
                loss = self._train_batch(X, y)
                losses[-1].append(loss)
            print(f'L={round(np.mean(losses[-1]), 4)}')
            if epoch % 5 == 0: # Just so I don't go insane waiting around
                torch.save(self.model.state_dict(), f'model/checkpoint.pth')
        return list(map(np.mean, losses))
                

    def evaluate(self, dataloader):
        self.model.eval() # Set model to evaluation mode
        metrics = []
        with torch.no_grad():
            for X, y, sub_id, split_id in dataloader:
                y_pred, y = self._forward_model(X, y)
                metrics.append(calculate_test_metrics(y_pred, y))
        return average_dictionary(metrics)