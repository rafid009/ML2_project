from turtle import color
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from data_module.bird_species_distribution import BirdSpeciesDataset
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from models.models import OccupancyDetectionModel
from tqdm import tqdm
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_val_test_dataset(dataset, val_split=0.20, test_split=0.20):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_split, shuffle=True, random_state=10)
    datasets = {}
    datasets['train_val'] = Subset(dataset, train_idx)
    datasets['test'] = Subset(dataset, test_idx)
    train_idx, val_idx = train_test_split(list(range(len(datasets['train_val']))), test_size=val_split, shuffle=True, random_state=50)
    datasets['val'] = Subset(datasets['train_val'], val_idx)
    datasets['train'] = Subset(datasets['train_val'], train_idx)
    datasets.pop('train_val')
    return datasets

tile_size = 64
data_root = '../bird_data'
batch_size = 2
occ_features = 5
detect_features = 3
n_epoch = 30
model_path = '../saved_bird_models'
dataset = BirdSpeciesDataset(data_root, tile_size)
datasets = train_val_test_dataset(dataset)

dataloaders = {x:DataLoader(datasets[x], batch_size=batch_size, shuffle=True) for x in ['train','val', 'test']}

# for i, sample in enumerate(dataloaders['train']):
#     print(f"{i}, {sample['occupancy_feature'].shape}\n{sample['detection_feature'].shape}\n{sample['detection'].shape}")
#     if i == 2:
#         break

model = OccupancyDetectionModel(occ_features, detect_features, 1).float()
model = model.to(device)
# criterion = nn.BCEWithLogitsLoss(reduction='mean')


def mask_out_nan(output, target):
    mask = ~torch.isnan(target)
    target = torch.nan_to_num(target)
    output = output * mask
    return output, target

def get_visit_likelihood(d, y):
    mask = ~torch.isnan(y)
    y_t = torch.nan_to_num(y)
    l = torch.pow(d, y_t) * torch.pow((1 - d), (1 - y_t))
    return l * mask, y_t

def get_avg_visit_loss(occ, likelihood, K_y):
    loss = torch.sum(torch.sum(occ * likelihood + (1 - occ) * K_y, dim=1), dim=1)
    return loss.mean()

def train(train_loader, val_loader, n_epoch, eval_path, n_visits=5):
    result_dict = {'train': [], 'val': []}
    model.train()
    for epoch in range(1, n_epoch + 1):
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}/{n_epoch}")
            total_train = 0
            total_val = 0
            count = 0
            avg_visit_loss = 0
            for data in train_loader:
                optimizer.zero_grad()
                likelihood_loss = torch.ones((batch_size, tile_size, tile_size))
                occ = None
                K_y = torch.zeros((batch_size, tile_size, tile_size))
                for v in range(n_visits):
                    output, occ, detect = model(data, v)
                    occ = torch.squeeze(occ)
                    detect = torch.squeeze(detect)
                    target = data[f'detection_{v}'].to(device)
                    bernouli_l, masked_y = get_visit_likelihood(detect, target)
                    likelihood_loss *= bernouli_l
                    K_y = torch.max(K_y, masked_y)
                loss = get_avg_visit_loss(occ, likelihood_loss, K_y)
                loss = loss / n_visits
                loss.backward()
                optimizer.step()
                val_loss = evaluate(val_loader)
                
                tepoch.set_postfix(train_loss=loss.item(), val_loss=val_loss)
                total_train += loss.item()
                total_val += val_loss
                count += 1
                tepoch.update(1)
            result_dict['train'].append(total_train/count)
            result_dict['val'].append(total_val/count)
        if epoch % 5 == 0:
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            torch.save(model.state_dict(), f"{model_path}/model-e{epoch}-l({np.round(val_loss, 5)}).pth")
    df = pd.DataFrame(result_dict)
    df.to_csv(eval_path, index=False)
                
def evaluate(val_loader, n_visits=5):
    total_loss = 0
    count = 0
    model.eval()
    for idx, data in enumerate(val_loader):
        avg_loss = 0
        avg_auc = 0
        likelihood_loss = torch.ones((batch_size, tile_size, tile_size))
        occ = None
        K_y = torch.zeros((batch_size, tile_size, tile_size))
        for v in range(n_visits):
            output, occ, detect = model(data, v)
            occ = torch.squeeze(occ)
            detect = torch.squeeze(detect)
            target = data[f'detection_{v}'].to(device)
            bernouli_l, masked_y = get_visit_likelihood(detect, target)
            likelihood_loss *= bernouli_l
            K_y = torch.max(K_y, masked_y)
        loss = get_avg_visit_loss(occ, likelihood_loss, K_y)
        total_loss += (loss / n_visits)
        count += 1
    model.train()
    return total_loss.item() / count

def plot_loss(n_epochs, train_losses, val_losses, lr):
    epochs = [e for e in range(1, n_epochs + 1)]
    plt.figure(figsize=(16,9))
    plt.title(f"Training vs validation cross entropy loss for lr={lr}", fontsize=20)
    plt.plot(epochs, train_losses, color='tab:red', label='Validation loss')
    plt.plot(epochs, val_losses, color='tab:orange', label='Training loss')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Number of epochs', fontsize=16)
    plt.ylabel('Cross entropy loss', fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig(f"plots/train-vs-val-plot-lr({lr}).png", dpi=300)
    plt.close()

lrs = [0.01, 0.001, 0.005, 0.0009]
plots_folder = '../plots_bird'
if not os.path.isdir(plots_folder):
    os.makedirs(plots_folder)
for lr in lrs:
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    plot_file = f"{plots_folder}/train-vs-valid-{lr}.csv"
    train(dataloaders['train'], dataloaders['val'], n_epoch, plot_file)
    df = pd.read_csv(plot_file)
    plot_loss(n_epoch, df['train'], df['val'], lr)

