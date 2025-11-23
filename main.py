import torch
from utils import tprint
from .model.FactorizePhys.FactorizePhys import FactorizePhys
from .model.MyFactorizePhys.MyFactorizePhys import MyFactorizePhys
from .model.MyFactorizePhys.MyFactorizePhysVector import MyFactorizePhysVector
from .training.loss import NegativePearsonCorrelationLoss
from .training.ModelTrainer import ModelTrainer


def train_model(model, model_name):
    tprint(f'Training', model_name)
    optimizer = NegativePearsonCorrelationLoss()
    loss_fn = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = ModelTrainer(model, optimizer, loss_fn)


def main():
    models = {
        'FP': FactorizePhys(),
        'MFP': MyFactorizePhys(),
        'MFPV': MyFactorizePhysVector()
    }


if __name__ == '__main__':
    main()