import torch
from torch import nn
import tqdm # Progress bar
    

class ModelTrainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)


    def _normalize_tensor(self, y):
        return (y - torch.mean(y)) / torch.std(y)


    def _forward_model(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        if len(y.shape) > 2: y = y[..., 0] # The authors say there is multi-signal data
        y = self._normalize_tensor(y)
        y_pred = self.model(X)
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
            tbar = tqdm(dataloader, ncols=80)
            losses.append([])
            for X, y in tbar: # Looping through batches
                tbar.set_description(f'Train epoch {epoch}')
                loss = self._train_batch(X, y)
                losses[-1].append(loss)
        return losses
                

    def evaluate(self, dataloader, func):
        self.model.eval() # Set model to evaluation mode
        outputs = []
        with torch.no_grad():
            for X, y in dataloader:
                y_pred, y = self._forward_model(X, y)
                outputs.append(func(y_pred, y))
        return outputs