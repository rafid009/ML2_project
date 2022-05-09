
from typing import Callable, Dict, Optional
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from data_module.bird_species_distribution import BirdSpeciesDataset
import numpy as np
import pandas as pd
import os


root = 'data/T64'
data_dict = {
    'occu[ancy_features': f'{root}/occupancy_features_64.npy',
    'detection_features': [f'{root}/detection_features_v1_64.npy', f'{root}/detection_features_v2_64.npy', 
                            f'{root}/detection_features_v3_64.npy', f'{root}/detection_features_v4_64.npy', 
                            f'{root}/detection_features_v5_64.npy'],
    'detection_label': f'{root}/detection_60_64.npy'
    }


occ = np.load(data_dict['occu[ancy_features'])
print(occ.shape)
detection_features = []
for d in data_dict['detection_features']:
    dfeat = np.load(d)
    # print(dfeat.shape)
    detection_features.append(dfeat)
detection = np.load(data_dict['detection_label'])
print(detection.shape)
dfeat_visits = np.stack(detection_features, axis=1)
print(dfeat_visits.shape)

di = {'a':[1,2], 'b':[2,3], 'c':[3,4]}
df = pd.DataFrame(di)
print(df)
df.to_csv('df.csv', index=False)
df1 = pd.read_csv('df.csv')
print(df1)