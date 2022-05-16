import torch
from data_module.bird_species_distribution import BirdSpeciesDataset
from torch.utils.data import DataLoader

tile_size = 64
data_root = '../bird_data'
data = BirdSpeciesDataset(data_root, tile_size)

dataloader = DataLoader(data, batch_size=10, shuffle=True)

for i, sample in enumerate(dataloader):
    # print(f"{i}, {sample['occupancy_feature'].shape}\n{sample['detection_feature'].shape}\n{sample['detection'].shape}")
    print(f'{i}:')
    for key in sample.keys():
        print(f"\t{key} = {sample[key].shape}")
    if i == 2:
        break

# DataLoader(transformed_dataset, batch_size=4,
#                         shuffle=True, num_workers=0)