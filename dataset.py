import pandas as pd
import numpy as np
from torch.utils.data import Dataset


def data_loader(drug1_chemicals, drug2_chemicals, cell_line_gex, comb_data_name):
    print("File reading ...")
    comb_data = pd.read_csv(comb_data_name, sep="\t")
    synergies = np.array(comb_data["synergy_loewe"], dtype=np.float32)
    cell_line = np.loadtxt(cell_line_gex, delimiter=',', dtype=np.float32)
    chem1 = np.loadtxt(drug1_chemicals, delimiter=',', dtype=np.float32)
    chem2 = np.loadtxt(drug2_chemicals, delimiter=',', dtype=np.float32)
    chem1 = np.nan_to_num(chem1)
    chem2 = np.nan_to_num(chem2)
    return chem1, chem2, cell_line, synergies


class Scaler:
    def __init__(self):
        self.std1 = None
        self.means1 = None
        self.feat_filt = None
        self.std2 = None
        self.means2 = None
        self.norm = None

    def fit_transform(self, X, norm='tanh_norm'):
        self.norm = norm
        self.std1 = np.nanstd(X, axis=0)
        self.feat_filt = self.std1!=0
        X = X[:, self.feat_filt]
        X = np.ascontiguousarray(X)
        self.means1 = np.mean(X, axis=0)
        X = (X - self.means1) / self.std1[self.feat_filt]
        if norm == 'norm':
            return X
        elif norm == 'tanh':
            return np.tanh(X)
        elif norm == 'tanh_norm':
            X = np.tanh(X)
            self.means2 = np.mean(X, axis=0)
            self.std2 = np.std(X, axis=0)
            X = (X - self.means2) / self.std2
            X[:, self.std2==0]=0
        return X

    def transform(self, X):
        X = X[:, self.feat_filt]
        X = np.ascontiguousarray(X)
        X = (X - self.means1) / self.std1[self.feat_filt]
        if self.norm == 'norm':
            return X
        elif self.norm == 'tanh':
            return np.tanh(X)
        elif self.norm == 'tanh_norm':
            X = np.tanh(X)
            X = (X - self.means2) / self.std2
            X[:, self.std2==0]=0
        return X


class MMDataset(Dataset):
    "MatchMaker Dataset."

    def __init__(self, cell_lines, chem1, chem2, synergies, index_file, train=False):
        indices = np.loadtxt(index_file, dtype=np.int)
        self.cells = cell_lines[indices]
        self.chem1 = chem1[indices]
        self.chem2 = chem2[indices]
        self.synergies = synergies[indices]
        self.train = train

    def calculate_weight(self):
        min_s = np.amin(self.synergies)
        self.loss_weight = np.log(self.synergies - min_s + np.e)

    def normalize(self, chem_scaler=None, cell_scaler=None):
        if chem_scaler is None:
            chem_scaler = Scaler() 
            self.chem1, self.chem2 = np.split(chem_scaler.fit_transform(np.concatenate([self.chem1, self.chem2], axis=0)), 2)
            # chem2_scaler = Scaler()
            # self.chem2 = chem2_scaler.fit_transform(self.chem2)
            cell_scaler = Scaler()
            self.cells = cell_scaler.fit_transform(self.cells)
            return chem_scaler, cell_scaler

        self.chem1, self.chem2 = np.split(chem_scaler.transform(np.concatenate([self.chem1, self.chem2], axis=0)),2)
        # self.chem2 = chem2_scaler.transform(self.chem2)
        self.cells = cell_scaler.transform(self.cells)
    

    def __len__(self):
        return self.cells.shape[0]

    def __getitem__(self, idx):
        sample = {
            'cell': self.cells[idx],
            'chem1': self.chem1[idx],
            'chem2': self.chem2[idx],
            'synergy': self.synergies[idx]
        }
        if self.train:
            sample['loss_weight'] = self.loss_weight[idx]
        return sample

