import os
import os.path as osp
import numpy as np
from torch.utils.data import Dataset
import json
import torch

def process_raw_data(root, processed_dir, out_path):
    data_files_dict = {
    'occupancy_features': f'{root}/occupancy_features_64.npy',
    'detection_features': [f'{root}/detection_features_v1_64.npy', f'{root}/detection_features_v2_64.npy', 
                            f'{root}/detection_features_v3_64.npy', f'{root}/detection_features_v4_64.npy', 
                            f'{root}/detection_features_v5_64.npy'],
    'detection_label': f'{root}/detection_60_64.npy'
    }
    processed_data_dict = {'occupancy_features': [], 'detection_features': [], 'detection_label': []}
    occ_features = np.load(data_files_dict['occupancy_features'])
    detect_visits = []
    label = np.load(data_files_dict['detection_label'])

    for d in data_files_dict['detection_features']:
        dfeat = np.load(d)
        detect_visits.append(dfeat)
    detect_features = np.stack(detect_visits, axis=1)

    for i in range(len(occ_features)):
        np.save(f"{processed_dir}/occ-feat-{i}.npy", occ_features[i])
        np.save(f"{processed_dir}/detect-label-{i}.npy", label[i])
        np.save(f"{processed_dir}/detect-fetures-{i}.npy", detect_features[i])
        processed_data_dict['occupancy_features'].append(f"{processed_dir}/occ-feat-{i}.npy")
        processed_data_dict['detection_features'].append(f"{processed_dir}/detect-fetures-{i}.npy")
        processed_data_dict['detection_label'].append(f"{processed_dir}/detect-label-{i}.npy")
    
    with open(out_path, 'w') as out:
        json.dump(processed_data_dict, out)
    return processed_data_dict


class BirdSpeciesDataset(Dataset):
    def __init__(self, data_root, tile_size): 
        self.tile_size = tile_size
        self.data_root = f"{data_root}/T{tile_size}"
        self.processed_meta_data = None
        self.processed_meta_path = f"{self.data_root}/processed_meta.json"
        self.processed_dir = f"{self.data_root}/processed/"

        if not osp.exists(self.processed_meta_path):
            if not osp.isdir(self.processed_dir):
                os.makedirs(self.processed_dir)
            self.processed_meta_data = process_raw_data(self.data_root, self.processed_dir, self.processed_meta_path)
        else:
            with open(self.processed_meta_path, 'r') as json_file:
                self.processed_meta_data = json.load(json_file)

    def __len__(self):
        return len(self.processed_meta_data['detection_label'])

    def __getitem__(self, idx):
        sample = {'occupancy_feature': None, 'detection_feature': None, 'detection': None}
        sample['occupancy_feature'] = torch.tensor(np.load(self.processed_meta_data['occupancy_features'][idx]), dtype=torch.double)
        sample['detection_feature'] = torch.tensor(np.load(self.processed_meta_data['detection_features'][idx]), dtype=torch.double)
        sample['detection'] = torch.tensor(np.load(self.processed_meta_data['detection_label'][idx]), dtype=torch.double)
        return sample