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
    return chem1, chem2, cell_line, synergies


class MMDataset(Dataset):
    "MatchMaker Dataset."

    def __init__(self, cell_lines, chem1, chem2, synergies, index_file):
        indices = np.loadtxt(index_file, dtype=np.int)
        self.cells = cell_lines[indices]
        self.chem1 = chem1[indices]
        self.chem2 = chem2[indices]
        self.synergies = synergies[indices]
    

    def __len__(self):
        return self.cells.shape[0]

    def __getitem__(self, idx):
        sample = {
            'cell': self.cells[idx],
            'chem1': self.chem1[idx],
            'chem2': self.chem2[idx],
            'synergy': self.synergies[idx]
        }
        return sample

