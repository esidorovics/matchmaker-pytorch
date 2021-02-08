import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np


class MatchMaker(nn.Module):
    
    def __init__(self, dsn_layers, spn_layers, drug_dim, cell_line_dim, in_drop_prob, drop_prob):
        super().__init__()
        dsn_layers.insert(0, drug_dim+cell_line_dim)
        dsn = [nn.BatchNorm1d(dsn_layers[0])]
        for i, size in enumerate(dsn_layers[1:], 1):
            dsn.append(nn.Linear(dsn_layers[i-1], size))
            if i < len(dsn_layers)-1 :
                dsn.append(nn.ReLU())
                if i == 1:
                    dsn.append(nn.Dropout(p=in_drop_prob))
                else:
                    dsn.append(nn.Dropout(p=drop_prob))
        self.dsn = nn.Sequential(*dsn)

        spn = []
        spn_layers.insert(0, dsn_layers[-1]*2)
        for i, size in enumerate(spn_layers[1:], 1):
            spn.extend([nn.Linear(spn_layers[i-1], size),
                        nn.ReLU()])
            if i == len(spn_layers)-1:
                spn.append(nn.Linear(size, 1))
            else:
                spn.append(nn.Dropout(drop_prob))
        self.spn = nn.Sequential(*spn)


    def forward(self, drug1, drug2, cell):
        x1 = torch.cat((drug1, cell), 1)
        x2 = torch.cat((drug2, cell), 1)
        embedding1 = self.dsn(x1)
        embedding2 = self.dsn(x2)
        combination = torch.cat((embedding1, embedding2), 1)
        out = self.spn(combination)
        return out.flatten()


def train(model, train_loader, valid_loader, log_file, epochs, patience, model_name, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_val_loss = float('inf')
    patience_level = 0

    with open(log_file, 'w') as f:
        f.write(f'Epoch\tTrain Loss\tValidation Loss\n')
    print(f'Epoch\tTrain Loss\tValidation Loss')

    for epoch in range(epochs):
        running_loss = 0.0
        for data in train_loader:
            optimizer.zero_grad()

            drug1 = data['chem1'].to(device)
            drug2 = data['chem2'].to(device)
            cell = data['cell'].to(device)
            synergy = data['synergy'].to(device)
            
            preds = model(drug1, drug2, cell)
            loss = criterion(synergy, preds)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            for data in valid_loader:
                drug1 = data['chem1'].to(device)
                drug2 = data['chem2'].to(device)
                cell = data['cell'].to(device)
                synergy = data['synergy'].to(device)

                preds = model(drug1, drug2, cell)
                loss = criterion(synergy, preds)
                val_loss += loss.item()
            model.train()
        patience_level += 1
        if best_val_loss > val_loss:
            print('new best Model')
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_name)
            patience_level = 0

        if patience_level > patience:
            break

        with open(log_file, 'a') as f:
            f.write(f'{epoch}\t{running_loss}\t{val_loss}\n')
        print(f'{epoch}\t{running_loss}\t{val_loss}')


def predict(model, test_loader, labels, device):
    model.eval()
    all_preds = None
    with torch.no_grad():
        val_loss = 0.0
        for data in test_loader:
            drug1 = data['chem1'].to(device)
            drug2 = data['chem2'].to(device)
            cell = data['cell'].to(device)
            synergy = data['synergy'].to(device)

            preds = model(drug1, drug2, cell).cpu().numpy()
            if all_preds is None:
                all_preds = preds
            else:
                all_preds = np.concatenate((all_preds, preds))

    return all_preds
