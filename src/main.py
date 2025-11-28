# Absolute imports
import torch
import os
import random
import numpy as np
import pandas as pd
from itertools import product
# Relative file imports
from .utils import tprint, tsince
from .config import _C as config
# Source module imports
from src.dataset import data_loader # This code is all from other researchers for loading their datasets
from src.model.FactorizePhys.FactorizePhys import FactorizePhys
from src.model.MyFactorizePhys.MyFactorizePhys import MyFactorizePhys
from src.model.MyFactorizePhys.MyFactorizePhysVector import MFP_NMF, MyFactorizePhysVector
from src.model.MyFactorizePhys.MFPVLoader import MFPVLoader
from src.training.loss import NegativePearsonCorrelationLoss
from src.training.ModelTrainer import ModelTrainer
from src.plot import plot_losses, plot_test_metrics


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)



MODELS = {
    'FP': FactorizePhys,
    'MFP': MyFactorizePhys,
    # 'MFPV': MyFactorizePhysVector
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
    'UBFC-rPPG': data_loader.UBFCrPPGLoader.UBFCrPPGLoader
}


DATADIRS = {
    'UBFC-rPPG': 'data/UBFC-rPPG_2'
}


MODEL_PATHS = {
    'FP': 'model/FP_50.pth',
    'MFP': 'model/MFP_50.pth'
}


def train_model(trainer, model_name, model, dataset_name, dataloader):
    start_time = tprint(f'Training', model_name, 'using', dataset_name)
    losses = trainer.train(dataloader, num_epochs=50)
    tprint('Finished training in', tsince(start_time))
    torch.save(model.state_dict(), f'model/{model_name}.pth')
    plot_losses(losses, 'r', model_name=model_name)


def load_model(model_name, *args, **kwargs):
    model = MODELS[model_name](*args, **kwargs)
    model_path = MODEL_PATHS.get(model_name)
    pretrained = model_path is not None and os.path.isfile(model_path)
    if pretrained:
        tprint(f'Loading pretrained model at {os.path.basename(model_path)}')
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)
    return model, pretrained


def create_MFPV(dname_vector, force_preproc=True):
    dataloader = MFPVLoader(dname_vector, config.TRAIN.DATA)
    if force_preproc or not os.path.isdir(dname_vector):
        MFP_model, MFP_pretrained = load_model('MFP')
        assert MFP_pretrained # Must be pretrained
        dataloader.preprocess(MFP_model)
    model, pretrained = load_model('MFPV', (160, 512, 512, 160))
    return model, pretrained, dataloader



def main():
    metrics_df = pd.DataFrame(columns='r value,RMSE,MAE,MAPE'.split(','))
    for model_name, dataset_name in product(MODELS, DATADIRS):
        if model_name == 'MFPV':
            model, pretrained, dataloader = create_MFPV('data/factorized')
        else:
            model, pretrained = load_model(model_name)
            datadir = os.path.abspath(DATADIRS[dataset_name])
            dataloader = DATALOADERS[dataset_name](dataset_name, datadir, config.TRAIN.DATA)
        train_dataloader, test_dataloader = torch.utils.data.random_split(dataloader, [0.8, 0.2])
        loss_fn = NegativePearsonCorrelationLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
        trainer = ModelTrainer(model, optimizer, loss_fn)
        if not pretrained:
            train_model(trainer, model_name, model, dataset_name, train_dataloader)
        metrics_df.loc[model_name] = trainer.evaluate(test_dataloader)
    plot_test_metrics(metrics_df)


if __name__ == '__main__':
    tprint('Running main.py...')
    main()