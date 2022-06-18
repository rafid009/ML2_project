from turtle import color
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from data_module.SDM_model import SpeciesDataset
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
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import json
import sys
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPSILON = 1e-7

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
data_root = '../sdm_data'
batch_size = 32
occ_features = 5
detect_features = 3
n_epoch = 20
model_path = '../sdm_models_sage_occt'
if not os.path.isdir(model_path):
    os.makedirs(model_path)
dataset = SpeciesDataset(data_root, tile_size)
datasets = train_val_test_dataset(dataset)

dataloaders = {x:DataLoader(datasets[x], batch_size=batch_size, shuffle=True) for x in ['train','val', 'test']}

if sys.argv[1] == 'graph':
    model = OccupancyDetectionModel(occ_features, detect_features, 1, is_graph=True).float()
else:
    model = OccupancyDetectionModel(occ_features, detect_features, 1, is_graph=False).float()
model = model.to(device)


def mask_out_nan(output, target):
    mask = ~torch.isnan(target)
    target = torch.nan_to_num(target)
    output = output * mask
    mask.detach()
    return output, target

def get_visit_likelihood(d, y):
    mask = ~torch.isnan(y).to(device)
    y_t = torch.nan_to_num(y).to(device)
    l = torch.pow(d, y_t) * torch.pow((1 - d), (1 - y_t))
    l = l.to(device)
    l = l * mask
    return l, y_t

def get_avg_visit_loss(occ, likelihood, K_y):
    l = occ * likelihood + (1 - occ) * K_y
    l = torch.flatten(l, start_dim=1) + EPSILON
    ll = torch.log(l)
    nll = -1.0 * torch.mean(ll, dim=1)
    loss = torch.mean(nll)
    return loss

def train(train_loader, val_loader, n_epoch, eval_path, lr, graph=None, n_visits=5):
    result_dict = {'train': [], 'val': []}
    model.train()
    auc_dict = {}
    for epoch in range(1, n_epoch + 1):
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}/{n_epoch}")
            total_train = 0
            total_val = 0
            count = 0
            for data in train_loader:
                optimizer.zero_grad()
                
                b_size = min(batch_size, len(data['occupancy_feature']))
                likelihood_loss = torch.ones((b_size, tile_size, tile_size), requires_grad=False).to(device)
                occ = None
                K_y = torch.zeros((b_size, tile_size, tile_size), requires_grad=False).to(device)
                for v in range(n_visits):
                    # print(f"d_target: {data[f'detection_{v}'].shape}")
                    output, occ, detect = model(data, v)
                    occ = torch.squeeze(occ)
                    detect = torch.squeeze(detect)
                    target = data[f'detection_{v}'].to(device)
                    with torch.no_grad():
                        bernouli_l, masked_y = get_visit_likelihood(detect, target)
                        # print(f"bernouli: {bernouli_l}")
                        likelihood_loss = likelihood_loss * bernouli_l
                        # print(f"likeli: {likelihood_loss}")
                        K_y = torch.max(K_y, masked_y)
                with torch.no_grad():
                    K_y = 1 - K_y
                loss = get_avg_visit_loss(occ, likelihood_loss, K_y)
                loss = loss / n_visits
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    #val_loss, auc_i = evaluate(val_loader)
                    val_loss = evaluate(val_loader)

                tepoch.set_postfix(train_loss=loss.item(), val_loss=val_loss)
                total_train += loss.item()
                total_val += val_loss
                count += 1
                tepoch.update(1)
                # exit(0)
            result_dict['train'].append(total_train/count)
            result_dict['val'].append(total_val/count)
        if epoch % 5 == 0:
            graph_dir = f'{model_path}/{graph}'
            if not os.path.isdir(graph_dir):
                os.makedirs(graph_dir)
            graph_dir = f'{graph_dir}/{lr}'
            if not os.path.isdir(graph_dir):
                os.makedirs(graph_dir)
            torch.save(model.state_dict(), f"{graph_dir}/model-{graph}-e{epoch}-{lr}-{np.round(val_loss, 5)}.pth")
    df = pd.DataFrame(result_dict)
    df.to_csv(eval_path, index=False)


def get_metrics (y_true, y_score):
   
    '''
        Takes flattened arrays of true labels and predicted scores 
        Returns area under ROC curve (AUROC) and area under Precision-Recall curved (AUPRC)
        Returns -1 if true labels only have one class. 
        Parameters
            y_true : ndarray of ground truth class labels
            y_score : ndarray of target scores
    
    '''
    y_true = y_true.flatten()
    y_score = y_score.flatten()

    
    # Check if there are any NaNs in the the class labels
    #   if present, remove those pixels
    
    nan_indices_true = np.argwhere(np.isnan(y_true)) # Get indices of NaN pixels
    nan_indices_score = np.argwhere(np.isnan(y_score))
    
    
    print (nan_indices_true, nan_indices_score )

    nan_indices = np.unique(np.concatenate((nan_indices_true, nan_indices_score)))

    #if(not np.any(nan_indices)): 
    #if NaNs are present they will be deleted, else the arrays will be unchanged
    
    print(y_true.shape, y_score.shape)
    print(y_true, y_score)
    y_true = np.delete(y_true, nan_indices)
    y_score = np.delete(y_score, nan_indices)
    print(y_true.shape, y_score.shape)
    print(y_true, y_score)


    # Check if only one class is present in class labels then scores are undefined 
    #   (we can only calculate tile-wise AUCs for tiles with atleast 1 pixel of each class in true label)
    #   if AUC is undefined skip tile and return -1 to indicate so
    
    print(nan_indices)
    print(np.unique(y_true))
    if(np.unique(y_true).shape[0]==1):
        return -1

    else:
        auroc = roc_auc_score(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auprc = auc(recall, precision)

        return {'AUROC': auroc, 'AUPRC': auprc} 
                
def evaluate(val_loader, n_visits=5):
    total_loss = 0
    count = 0
    model.eval()
    #auroc_occ_dict = {}
    #auprc_occ_dict = {}
    #auroc_det_dict = {}
    #auprc_det_dict = {}
    for idx, data in enumerate(val_loader):
        avg_loss = 0
      #  avg_auroc_v = 0
       # avg_auprc_v = 0
        b_size = min(batch_size, len(data['occupancy_feature']))
        likelihood_loss = torch.ones((b_size, tile_size, tile_size)).to(device)
        occ = None
        occ_target = None
        K_y = torch.zeros((b_size, tile_size, tile_size)).to(device)
        for v in range(n_visits):
            output, occ, detect = model(data, v)
            occ = torch.squeeze(occ)
            occ_target = data[f'occupancy_label'].to(device)
            detect = torch.squeeze(detect)
            target = data[f'detection_{v}'].to(device)
            bernouli_l, masked_y = get_visit_likelihood(detect, target)
            # print(f"det: {detect.shape}, targ: {target.shape}, bernou: {bernouli_l.shape}, likeli: {likelihood_loss.shape}")
            likelihood_loss = likelihood_loss * bernouli_l

        
            K_y = torch.max(K_y, masked_y)
     

        K_y = 1 - K_y
    
        #avg_auroc_v = avg_auroc_v/n_visits
        #avg_auprc_v = avg_auprc_v/n_visits

        #auroc_det_dict[idx] = avg_auroc_v
        #auprc_det_dict[idx] = avg_auprc_v

        loss = get_avg_visit_loss(occ, likelihood_loss, K_y)
        total_loss += (loss / n_visits)
        count += 1

    model.train()
    return total_loss.item() / count


def test(test_loader, n_visits=5):
    model.eval()
    auroc_occ_dict = {}
    auprc_occ_dict = {}
    auroc_det_dict = {}
    auprc_det_dict = {}
    for idx, data in enumerate(test_loader):
        avg_auroc_v = 0
        avg_auprc_v = 0
        b_size = min(batch_size, len(data['occupancy_feature']))
        occ = None
        occ_target = None
        for v in range(n_visits):
            output, occ, detect = model(data, v)
            occ = torch.squeeze(occ)
            occ_target = data[f'occupancy_label'].to(device)
            detect = torch.squeeze(detect)
            target = data[f'detection_{v}'].to(device)
            # print(f"det: {detect.shape}, targ: {target.shape}, bernou: {bernouli_l.shape}, likeli: {likelihood_loss.shape}")
            
            output = torch.flatten(output, start_dim=1).cpu().detach().numpy()
            target = torch.flatten(target, start_dim=1).cpu().detach().numpy()
        
            det_metrics = get_metrics(target, output)
        
            if(det_metrics != -1):
                avg_auroc_v += det_metrics['AUROC'] 
                avg_auprc_v += det_metrics['AUPRC']
        
     
        occ = torch.flatten(occ, start_dim=1).cpu().detach().numpy()
        occ_target = torch.flatten(occ_target, start_dim=1).cpu().detach().numpy()
    
        occ_metrics = get_metrics(occ_target, occ)
        
        if(occ_metrics != -1):
            auroc_occ_dict[idx] = occ_metrics['AUROC']
            auprc_occ_dict[idx] = occ_metrics['AUPRC']

    
        avg_auroc_v = avg_auroc_v/n_visits
        avg_auprc_v = avg_auprc_v/n_visits

        auroc_det_dict[idx] = avg_auroc_v
        auprc_det_dict[idx] = avg_auprc_v


    auc_dict = {'OCC-AUROC': auroc_occ_dict,
                'OCC-AUPRC': auprc_occ_dict, 
                'DET-AUROC': auroc_det_dict,
                'DET-AUPRC': auprc_det_dict
        }
    # model.train()
    return auc_dict

def plot_loss(n_epochs, train_losses, val_losses, lr, plots_folder):
    epochs = [e for e in range(1, n_epochs + 1)]
    plt.figure(figsize=(16,9))
    plt.title(f"Training vs validation cross entropy loss for lr={lr}", fontsize=20)
    plt.plot(epochs, train_losses, color='tab:red', label='Validation loss')
    plt.plot(epochs, val_losses, color='tab:orange', label='Training loss')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Number of epochs', fontsize=18)
    plt.ylabel('Cross entropy loss', fontsize=18)
    plt.legend(fontsize=18)
    plt.savefig(f"{plots_folder}/train-vs-val-plot-lr({lr}).png", dpi=300)
    plt.close()


lrs = [0.001]#[0.01, 0.001, 0.1, 0.05]
plots_folder = '../SDM_plots_sage_occt'
graphs = ['gcn', 'sage', 'gat', 'gat2', 'supgat', 'none']

if not os.path.isdir(plots_folder):
    os.makedirs(plots_folder)
for lr in lrs:
    for g in graphs:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        plot_file = f"{plots_folder}/{g}-train-vs-valid-{lr}.csv"
        if g != 'none':
            train(dataloaders['train'], dataloaders['val'], n_epoch, plot_file, lr, graph=g)
        else:
            train(dataloaders['train'], dataloaders['val'], n_epoch, plot_file, lr)
        df = pd.read_csv(plot_file)
        plot_loss(n_epoch, df['train'], df['val'], lr, plots_folder)
        aucs = test(dataloaders['test'])

        with open(f"{plots_folder}/{g}-auc-test-{lr}.json", "w") as outfile:
            json.dump(aucs, outfile)
