import os
import os.path as osp
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import json
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def process_raw_data(root, processed_dir, out_path, k=3, w_s=0.5, w_e=1.0):
    data_files_dict = {
    'occupancy_features': f'{root}/occupancy_features_64.npy',
    'detection_features': [f'{root}/detection_features_v1_64.npy', f'{root}/detection_features_v2_64.npy', 
                            f'{root}/detection_features_v3_64.npy', f'{root}/detection_features_v4_64.npy', 
                            f'{root}/detection_features_v5_64.npy'],
    'occupancy_label': f'{root}/occupancy_true_64.npy',
    'detection_label': f'{root}/detection_60_64.npy'
    }
 
    processed_data_dict = {'occupancy_features': [], 'detection_features': [], 'detection_label': [], 'neighbors': [], 'occupancy_label': []}
    occ_features = np.load(data_files_dict['occupancy_features'])
    detect_visits = []
    label = np.load(data_files_dict['detection_label'])
    occ_label = np.load(data_files_dict['occupancy_label'])

    for d in data_files_dict['detection_features']:
        dfeat = np.load(d)
        detect_visits.append(dfeat)
    detect_features = np.stack(detect_visits, axis=1)

    for i in tqdm(range(len(occ_features)), desc='Data Saved'):
        if np.sum(~np.isnan(occ_features[i])) == 0 or np.sum(~np.isnan(detect_features[i])) == 0:
            continue
        if np.sum(np.isnan(occ_features[i])) > 0:
            occ_features[i] = np.nan_to_num(occ_features[i])
        if np.sum(np.isnan(detect_features[i])) > 0:
            detect_features[i] = np.nan_to_num(detect_features[i], nan=-1)
        neighbors = save_k_neighbors(torch.tensor(occ_features[i], dtype=torch.float32), k, w_s, w_e)
        np.save(f"{processed_dir}/neighbors_{i}_{k}_{w_s}_{w_e}.npy", neighbors)
        np.save(f"{processed_dir}/occ-feat-{i}_{k}_{w_s}_{w_e}.npy", occ_features[i])
        np.save(f"{processed_dir}/detect-label-{i}_{k}_{w_s}_{w_e}.npy", label[i])
        np.save(f"{processed_dir}/detect-fetures-{i}_{k}_{w_s}_{w_e}.npy", detect_features[i])
        np.save(f"{processed_dir}/occ-true-{i}_{k}_{w_s}_{w_e}.npy", occ_label[i]) 
        processed_data_dict['occupancy_features'].append(f"{processed_dir}/occ-feat-{i}_{k}_{w_s}_{w_e}.npy")
        processed_data_dict['detection_features'].append(f"{processed_dir}/detect-fetures-{i}_{k}_{w_s}_{w_e}.npy")
        processed_data_dict['detection_label'].append(f"{processed_dir}/detect-label-{i}_{k}_{w_s}_{w_e}.npy")
        processed_data_dict['neighbors'].append(f"{processed_dir}/neighbors_{i}_{k}_{w_s}_{w_e}.npy")
        processed_data_dict['occupancy_label'].append(f"{processed_dir}/occ-true-{i}_{k}_{w_s}_{w_e}.npy")

    with open(out_path, 'w') as out:
        json.dump(processed_data_dict, out)
    return processed_data_dict

def save_k_neighbors(occ_f, k, w_s, w_e):
    x = torch.arange(start=0, end=occ_f.shape[2], dtype=torch.float32)
    xx = x.repeat([occ_f.shape[1], 1])
    y = torch.arange(start=0, end=occ_f.shape[1], dtype=torch.float32).view(-1, 1)
    yy = y.repeat([1, occ_f.shape[2]])
    # pos = w_s * torch.stack([xx, yy], dim=0)
    pos = torch.stack([xx, yy], dim=0)
    # occ_f = w_e * occ_f
    occ = torch.cat([occ_f, pos], axis=0)
    occ = torch.permute(occ, (1,2,0))
    occ = occ.flatten(start_dim=0, end_dim=1)
    occ_normalizer = StandardScaler()
    occ_norm = occ_normalizer.fit_transform(occ)
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(occ_norm)
    neighbor_edges = [[],[]]
    for i in range(len(occ)):
        occ_data = torch.unsqueeze(occ[i], dim=0)
        dists, neighs = knn.kneighbors(occ_data)
        # print(f"dist: {dists.shape}, neighs: {neighs.shape}")
        neighbors = neighs[0].tolist()
        # print(neighbors)
        for n in neighbors:
            neighbor_edges[0].append(i)
            neighbor_edges[1].append(n)
            neighbor_edges[0].append(n)
            neighbor_edges[1].append(i)
    print(f"neighbor len: {len(neighbor_edges)}, n[0]: {len(neighbor_edges[0])}, n[1]: {len(neighbor_edges[1])}")
    nei = np.array(neighbor_edges)
    return nei

class SpeciesDataset(Dataset):
    def __init__(self, data_root, tile_size, n_visits=5, k=3, reload=False, w_s=0.5, w_e=1.0): 
        self.tile_size = tile_size
        self.data_root = f"{data_root}/T{tile_size}"
        self.processed_meta_data = None
        self.processed_meta_path = f"{self.data_root}/processed_meta.json"
        self.processed_dir = f"{self.data_root}/processed/"
        self.n_visits = n_visits
        self.k = k
        self.reload = reload

        if self.reload or not osp.exists(self.processed_meta_path):
            if not osp.isdir(self.processed_dir):
                os.makedirs(self.processed_dir)
            self.processed_meta_data = process_raw_data(self.data_root, self.processed_dir, self.processed_meta_path, self.k, w_s, w_e)
        else:
            with open(self.processed_meta_path, 'r') as json_file:
                self.processed_meta_data = json.load(json_file)

    def __len__(self):
        return len(self.processed_meta_data['detection_label'])

    def __getitem__(self, idx):
        sample = {}
        sample['occupancy_feature'] = torch.tensor(np.load(self.processed_meta_data['occupancy_features'][idx]), dtype=torch.float32)
        sample['neighbors'] = torch.tensor(np.load(self.processed_meta_data['neighbors'][idx], allow_pickle=True), dtype=torch.long)
        d_feats = torch.tensor(np.load(self.processed_meta_data['detection_features'][idx]), dtype=torch.float32)
        detections = torch.tensor(np.load(self.processed_meta_data['detection_label'][idx]), dtype=torch.float32)
        sample['occupancy_label'] = torch.tensor(np.load(self.processed_meta_data['occupancy_label'][idx]), dtype=torch.float32)  
        for v in range(self.n_visits):
            sample[f'detection_feature_{v}'] = d_feats[v]
            sample[f'detection_{v}'] = detections[v]
        return sample
