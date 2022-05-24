import torch
from data_module.bird_species_distribution import BirdSpeciesDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

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
dataset = BirdSpeciesDataset(data_root, tile_size)

datasets = train_val_test_dataset(dataset)

dataloaders = {x:DataLoader(datasets[x], batch_size=2, shuffle=True) for x in ['train','val', 'test']}

for i, sample in enumerate(dataloaders['train']):
    # print(f"{i}, {sample['occupancy_feature'].shape}\n{sample['detection_feature'].shape}\n{sample['detection'].shape}")
    print(f'{i}:')
    # print(f"{sample[f'detection_{0}'].shape}")
    for key in sample.keys():
        print(f"\t{key} = {sample[key].shape}")
    if i == 2:
        break

# DataLoader(transformed_dataset, batch_size=4,
#                         shuffle=True, num_workers=0)