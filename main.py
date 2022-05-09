from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from data_module.bird_species_distribution import BirdSpeciesDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from models.models import OccupancyDetectionModel
from tqdm import tqdm
import time

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
data_root = './data'
batch_size = 32
occ_features = 5
detect_feature_visits = 5
detect_features = 3
n_epoch = 100
dataset = BirdSpeciesDataset(data_root, tile_size)
datasets = train_val_test_dataset(dataset)

dataloaders = {x:DataLoader(datasets[x], batch_size=batch_size, shuffle=True) for x in ['train','val', 'test']}

# for i, sample in enumerate(dataloaders['train']):
#     print(f"{i}, {sample['occupancy_feature'].shape}\n{sample['detection_feature'].shape}\n{sample['detection'].shape}")
#     if i == 2:
#         break

model = OccupancyDetectionModel(occ_features, detect_feature_visits, 1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()


def train(train_loader, val_loader, n_epoch):
    
    model.train()
    for epoch in range(1, n_epoch + 1):
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for idx, data in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, data['detection'])
                loss.backward()
                optimizer.step()
                val_loss = evaluate(val_loader)
                tepoch.set_postfix(train_loss=loss.item(), val_loss=val_loss)
                time.sleep(0.1)
                
def evaluate(val_loader):
    total_loss = 0
    count = 0
    model.eval()
    for idx, data in enumerate(val_loader):
        output = model(data)
        loss = criterion(output, data['detection'])
        total_loss += loss.item()
        count += 1
    return total_loss / count