import os
import numpy as np
from dataset import data_loader, MMDataset
from torch.utils.data import DataLoader
from model import MatchMaker, train, predict
import argparse
import torch
import pandas as pd
import performance_metrics
import datetime

def main(args, device):
    os.makedirs('logs', exist_ok=True)

    chem1, chem2, cell_line, synergies = data_loader(args.drug1_chemicals, args.drug2_chemicals,
                                                    args.cell_line_gex, args.comb_data_name)

    architecture = pd.read_csv('architecture.txt')
    layers = {}
    layers['DSN'] = list(map(int, architecture['DSN_1'][0].split('-'))) # layers of Drug Synergy Network 1
    layers['SPN'] = list(map(int, architecture['SPN'][0].split('-'))) # layers of Synergy Prediction Network

    mm = MatchMaker(layers['DSN'], layers['SPN'], chem1.shape[1], cell_line.shape[1], 
                    args.in_drop, args.dropout)
    mm.to(device)

    if args.train_test_mode == 'train':
        #TODO remove features where there is no variation
        train_dataset = MMDataset(cell_line, chem1, chem2, synergies, args.train_ind)
        valid_dataset = MMDataset(cell_line, chem1, chem2, synergies, args.val_ind)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True, num_workers=0)
        start = datetime.datetime.now()
        print(f'Training starts at {start}')
        train(mm, train_loader, valid_loader, os.path.join('logs', args.train_log), args.epoch, 
              args.patience, args.model_name, device)
        print(f'Train time {datetime.datetime.now()-start}')

    mm.load_state_dict(torch.load(args.model_name))
    mm.eval()
    test_dataset = MMDataset(cell_line, chem1, chem2, synergies, args.test_ind)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    all_preds = predict(mm, test_loader, test_dataset.synergies, device)

    mse_value = performance_metrics.mse(test_dataset.synergies, all_preds)
    spearman_value = performance_metrics.spearman(test_dataset.synergies, all_preds)
    pearson_value = performance_metrics.pearson(test_dataset.synergies, all_preds)

    np.savetxt("logs/preds.txt", all_preds, delimiter=",")
    np.savetxt("logs/y_test.txt", test_dataset.synergies, delimiter=",")
    with open('logs/results.txt', 'w') as f:
        f.write(f"Pearson correlation: {pearson_value}")
        f.write(f"Spearman correlation: {spearman_value}")
        f.write(f"Mean squared error: {mse_value}")



torch.manual_seed(0)

parser = argparse.ArgumentParser(description='REQUEST REQUIRED PARAMETERS OF MatchMaker')

parser.add_argument('--comb-data-name', default='data/DrugCombinationData.tsv',
                    help="Name of the drug combination data")

parser.add_argument('--cell_line-gex', default='data/cell_line_gex.csv',
                    help="Name of the cell line gene expression data")

parser.add_argument('--drug1-chemicals', default='data/drug1_chem.csv',
                    help="Name of the chemical features data for drug 1")

parser.add_argument('--drug2-chemicals', default='data/drug2_chem.csv',
                    help="Name of the chemical features data for drug 2")

parser.add_argument('--train-test-mode', default='train', type = str,
                    help="Test or train mode (test/train)")

parser.add_argument('--train-ind', default='data/train_inds.txt',
                    help="Data indices that will be used for training")

parser.add_argument('--val-ind', default='data/val_inds.txt',
                    help="Data indices that will be used for validation")

parser.add_argument('--test-ind', default='data/test_inds.txt',
                    help="Data indices that will be used for test")

parser.add_argument('--model_name', default="matchmaker.pt",
                    help='Model name to save weights')

parser.add_argument('--in-drop', default=0.2, type=float,
                    help='Dropout probability for input layer')

parser.add_argument('--dropout', default=0.5, type=float,
                    help='Dropout proability for the network')

parser.add_argument('--epoch', default=1000, type=int,
                    help='Number of epochs')

parser.add_argument('--patience', default=100, type=int,
                    help='Patience variable')

parser.add_argument('--train-log', default='train.log', type=str,
                    help='Log file to write train and validation losses during the training')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available()
                               else "cpu")

main(args, device)
