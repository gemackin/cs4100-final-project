from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import re
import os
import torch
from src.model.MyFactorizePhys.MyFactorizePhysVector import MFP_NMF
from src.utils import tprint


class MFPVLoader(Dataset):
    def __init__(self, dname_vector, config):
        super().__init__()
        self.dname_vector = dname_vector
        self.dname_matrix = config.CACHED_PATH
        self.fname_list = config.FILE_LIST_PATH
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert config.DATA_FORMAT == 'NCDHW'
        self.load_data()

    def preprocess(self, MFP_model):
        tprint('Preprocessing MFPV inputs')
        os.makedirs(self.dname_vector, exist_ok=True)
        NMF_model = MFP_NMF(MFP_model)
        for fname_m, fname_v in zip(self.m_inputs, self.v_inputs):
            V = np.load(fname_m)
            V = np.float32(np.transpose(V, (3, 0, 1, 2)))
            V = torch.tensor(V, requires_grad=True).to(self.device)
            V = V.view((1, *V.shape)) # Accounting for funky stuff with FactorizePhys
            W, H = NMF_model(V) # Temporal, spatial vectors
            np.save(fname_v, W) # The actual data for analysis
            fname_spatial = fname_v.replace('input', 'spatial')
            np.save(fname_spatial, H) # Not important for anything

    def load_data(self):
        self.m_inputs = pd.read_csv(self.fname_list)['input_files'].tolist()
        self.labels = [f.replace('input', 'label') for f in self.m_inputs]
        self.v_inputs = [f.replace(self.dname_matrix, self.dname_vector) for f in self.m_inputs]
    
    def __len__(self):
        return len(self.v_inputs)
    
    def __getitem__(self, index):
        vector = np.float32(np.load(self.v_inputs[index]))
        label = np.float32(np.load(self.labels[index]))
        rgx = re.search('subject(\d+)_input(\d+)', self.v_inputs[index])
        return vector, label, int(rgx[1]), int(rgx[2]) # W, BVP, subject, split