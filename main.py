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
import re
import logging
import sys


def create_logger(name, save_dir, quiet = False):
    if name in logging.root.manager.loggerDict:
        return logging.getLogger(name)

    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger

def main(args, device):
    logger = create_logger(args.train_test_mode, args.save_dir)
    if logger is not None:
        debug, info = logger.debug, logger.info
    debug('Command line')
    debug(f'python {" ".join(sys.argv)}')
    debug('Args')
    debug(args)

    chem1, chem2, cell_line, synergies = data_loader(args.drug1_chemicals, args.drug2_chemicals,
                                                    args.cell_line_gex, args.comb_data_name)
    print(chem1.shape)
    print(chem1.max())
    print(chem1.min())
    print(chem2.shape)
    print(chem2.max())
    print(chem2.min())
    print(cell_line.shape)

    architecture = pd.read_csv('architecture.txt')
    layers = {}
    layers['DSN'] = list(map(int, architecture['DSN_1'][0].split('-'))) # layers of Drug Synergy Network 1
    layers['SPN'] = list(map(int, architecture['SPN'][0].split('-'))) # layers of Synergy Prediction Network

    # mm = MatchMaker(layers['DSN'], layers['SPN'], chem1.shape[1], cell_line.shape[1], 
    #                 args.in_drop, args.dropout)
    # mm.to(device)

    if args.train_test_mode == 'train':
        #TODO remove features where there is no variation
        train_dataset = MMDataset(cell_line, chem1, chem2, synergies, args.train_ind, train=True)
        chem_scaler, cell_scaler = train_dataset.normalize()
        train_dataset.calculate_weight()
        valid_dataset = MMDataset(cell_line, chem1, chem2, synergies, args.val_ind)
        valid_dataset.normalize(chem_scaler, cell_scaler)
        debug(f'Train set: {len(train_dataset)}')
        debug(f'Validation set: {len(valid_dataset)}')
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True, num_workers=0)

        drug_dim  = train_dataset.chem1.shape[1]
        cell_dim = train_dataset.cells.shape[1]
        mm = MatchMaker(layers['DSN'], layers['SPN'], drug_dim, cell_dim, args.in_drop, args.dropout)
        debug(mm)
        mm.to(device)

        start = datetime.datetime.now()
        debug(f'Training starts at {start}')
        train(mm, train_loader, valid_loader, logger, args.epoch, 
              args.patience, args.model_name, args.save_dir, device)
        debug(f'Train time {datetime.datetime.now()-start}')

    mm.load_state_dict(torch.load(os.path.join(args.save_dir, args.model_name)))
    mm.eval()
    test_dataset = MMDataset(cell_line, chem1, chem2, synergies, args.test_ind)
    test_dataset.normalize(chem_scaler, cell_scaler)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    debug(f'Test set: {len(test_dataset)}')

    all_preds = predict(mm, test_loader, device)

    mse_value = performance_metrics.mse(test_dataset.synergies, all_preds, debug)
    spearman_value = performance_metrics.spearman(test_dataset.synergies, all_preds, debug)
    pearson_value = performance_metrics.pearson(test_dataset.synergies, all_preds, debug)

    df = pd.DataFrame()
    df['preds'] = all_preds
    df['loewe'] = test_dataset.synergies
    df.to_csv(os.path.join(args.save_dir, 'mm-pytorch-preds.csv'), index=False)
    info(f"Pearson correlation: {pearson_value}")
    info(f"Spearman correlation: {spearman_value}")
    info(f"Mean squared error: {mse_value}")



torch.manual_seed(0)

parser = argparse.ArgumentParser(description='REQUEST REQUIRED PARAMETERS OF MatchMaker')

parser.add_argument('--comb-data-name', default='../matchmaker/data/v15/DrugCombinationData.tsv',
                    help="Name of the drug combination data")

parser.add_argument('--cell_line-gex', default='../matchmaker/data/v15/cell_line_gex.csv',
                    help="Name of the cell line gene expression data")

parser.add_argument('--drug1-chemicals', default='../matchmaker/data/v15/drug1_chem_v15.csv',
                    help="Name of the chemical features data for drug 1")

parser.add_argument('--drug2-chemicals', default='../matchmaker/data/v15/drug2_chem_v15.csv',
                    help="Name of the chemical features data for drug 2")

parser.add_argument('--train-test-mode', default='train', type = str,
                    help="Test or train mode (test/train)")

parser.add_argument('--train-ind', default='../matchmaker/data/v15/train_inds.txt',
                    help="Data indices that will be used for training")

parser.add_argument('--val-ind', default='../matchmaker/data/v15/val_inds.txt',
                    help="Data indices that will be used for validation")

parser.add_argument('--test-ind', default='../matchmaker/data/v15/test_inds.txt',
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

parser.add_argument('--result-dir', default='results', type=str,
                    help='Directory to store the results')


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available()
                               else "cpu")

    os.makedirs(args.result_dir, exist_ok=True)

    dir_names = [name for name in os.listdir(args.result_dir) 
                if os.path.isdir(os.path.join(args.result_dir, name))]

    iter = [re.findall(r'\d+', dname)[0] for dname in dir_names if re.findall(r'\d+', dname)]
    if len(iter) > 0:
        n = max(list(map(int, iter)))+1
    else:
        n = 0

    args.save_dir = os.path.join(args.result_dir, str(n).zfill(3)+'-'+args.train_test_mode)
    os.makedirs(args.save_dir)

    main(args, device)
