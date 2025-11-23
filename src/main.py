import torch
from utils import tprint, tsince
from .dataset import data_loader # This code is all from other researchers for loading their datasets
from .model.FactorizePhys.FactorizePhys import FactorizePhys
from .model.MyFactorizePhys.MyFactorizePhys import MyFactorizePhys
from .model.MyFactorizePhys.MyFactorizePhysVector import MyFactorizePhysVector
from .training.loss import NegativePearsonCorrelationLoss
from .training.ModelTrainer import ModelTrainer


MODELS = {
    'FP': FactorizePhys,
    'MFP': MyFactorizePhys,
    'MFPV': MyFactorizePhysVector
}


DATALOADERS = {
    # 'BP4DPlus': data_loader.BP4DPlusLoader.BP4DPlusLoader, # Need to contact authors
    # 'BP4DPlusBigSmall': data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader, # Need to contact authors
    # 'COHFACE': data_loader.COHFACELoader.COHFaceLoader, # Need to contact authors
    # 'iBVP': data_loader.iBVPLoader.iBVPLoader, # Need to contact authors
    # 'MMPD': data_loader.MMPDLoader.MMPDLoader, # Need to contact authors
    # 'PURE': data_loader.PURELoader.PURELoader, # Need to contact authors
    # 'SCAMPS': data_loader.SCAMPSLoader.SCAMPSLoader, # Synthetic videos created using Blender?
    # 'UBFC-PHYS': data_loader.UBFCPHYSLoader.UBFCPHYSLoader, # Need to create IEEE account
    'UBFC-rPPG_1': data_loader.UBFCrPPGLoader.UBFCrPPGLoader,
    'UBFC-rPPG_2': data_loader.UBFCrPPGLoader.UBFCrPPGLoader
}


def train_model(model_name, model, dataset_name, dataloader):
    start_time = tprint(f'Training', model_name, 'using', dataset_name)
    optimizer = NegativePearsonCorrelationLoss()
    loss_fn = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = ModelTrainer(model, optimizer, loss_fn)
    trainer.train(dataloader['train'], num_epochs=5)
    tprint('Finished training in', tsince(start_time))
    # outputs = trainer.evaluate(dataloader['test'])


def main():
    pass


if __name__ == '__main__':
    main()