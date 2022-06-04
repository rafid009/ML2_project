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


def save_test(test_loader, n_visits=5):
    model.eval()
   # auroc_occ_dict = {}
   # auprc_occ_dict = {}
   # auroc_det_dict = {}
   # auprc_det_dict = {}
    for idx, data in enumerate(test_loader):
    #    avg_auroc_v = 0
    #    avg_auprc_v = 0
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
            
            np.savetxt(f'{save_path}det_pred_{idx}_v{v}.csv', output, delimiter=',') 
            np.savetxt(f'{save_path}det_true_{idx}_v{v}.csv', target, delimiter=',')

           # det_metrics = get_metrics(target, output)
        
           # if(det_metrics != -1):
           #     avg_auroc_v += det_metrics['AUROC'] 
           #     avg_auprc_v += det_metrics['AUPRC']
        
     
        occ = torch.flatten(occ, start_dim=1).cpu().detach().numpy()
        occ_target = torch.flatten(occ_target, start_dim=1).cpu().detach().numpy()
    
        np.savetxt(f'{save_path}occ_pred_{idx}.csv', occ, delimiter=',') 
        np.savetxt(f'{save_path}occ_true_{idx}.csv', occ_target, delimiter=',')

       # occ_metrics = get_metrics(occ_target, occ)
        
       # if(occ_metrics != -1):
       #     auroc_occ_dict[idx] = occ_metrics['AUROC']
       #     auprc_occ_dict[idx] = occ_metrics['AUPRC']

    
       # avg_auroc_v = avg_auroc_v/n_visits
       # avg_auprc_v = avg_auprc_v/n_visits

       # auroc_det_dict[idx] = avg_auroc_v
       # auprc_det_dict[idx] = avg_auprc_v


    #auc_dict = {'OCC-AUROC': auroc_occ_dict,
    #            'OCC-AUPRC': auprc_occ_dict, 
    #            'DET-AUROC': auroc_det_dict,
    #            'DET-AUPRC': auprc_det_dict
    #    }
    # model.train()
    #return auc_dict
 
 
tile_size = 64
data_root = '../sdm_data'
batch_size = 32
occ_features = 5
detect_features = 3
n_epoch = 40
model_path = '../sdm_models_debug/'

if not os.path.isdir(model_path):
    os.makedirs(model_path)
dataset = SpeciesDataset(data_root, tile_size)
datasets = train_val_test_dataset(dataset)
 
dataloaders = {x:DataLoader(datasets[x], batch_size=batch_size, shuffle=True) for x in ['train','val', 'test']}
model_filename = sys.argv[2]
if sys.argv[1] == 'graph':
    model = OccupancyDetectionModel(occ_features, detect_features, 1, is_graph=True).float()
else:
    model = OccupancyDetectionModel(occ_features, detect_features, 1, is_graph=False).float()
model = model.to(device)

model.load_state_dict(torch.load(model_path+model_filename, map_location=torch.device(device)))
 
model.eval()

save_path = '../saved_predictions/'+model_filename+'/'
if not os.path.isdir(save_path):
    os.makedirs(save_path)


auc = save_test(dataloaders['test'])
