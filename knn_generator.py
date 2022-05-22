from scipy.fftpack import tilbert
import torch
from data_module.bird_species_distribution import BirdSpeciesDataset
from torch.utils.data import DataLoader
import numpy as np
import json
from tqdm import tqdm

#uses the top left of 
def get_euc_dist(loc1, loc2, tile_size):
    #to make the tile have total area of 1/sqrt(2) x 1/sqrt(2).
    #and account for only using the top left of each tile
    normalizer = 1 / (tile_size - 1)
    normalizer *= 1 / np.sqrt(2) #ensure maximum distance away is 1
    row1 = (loc1 // tile_size) * normalizer
    row2 = (loc2 // tile_size) * normalizer
    col1 = (loc1 % tile_size) * normalizer
    col2 = (loc2 % tile_size) * normalizer
    
    dist = np.sqrt((row2 - row1)**2 + (col2 - col1)**2)
    return dist



def get_distance(site1, idx_1, site2, idx_2, tile_size):
    euc_dist = get_euc_dist(idx_1, idx_2, tile_size)
    feature_dist = 0
    for i in range(len(site1)): #iterate each feature
        feature_dist += np.abs(site1[i].item() - site2[i].item())
    dist = euc_dist + feature_dist
    return dist


def get_neighbors(cur_site, cur_site_idx, tile, k):
    #fill out distance array
    dist_arr = np.full(tile_size * tile_size, 0) #tile-sized array
    new_site_idx = 0
    for new_site in tile:
        if(new_site_idx == cur_site_idx):
            dist_arr[new_site_idx] = 100 #a site can't be its own nearest neighbor
        else:
            dist = get_distance(cur_site, cur_site_idx, new_site, new_site_idx, tile_size)
            dist_arr[new_site_idx] = dist
        new_site_idx += 1

    #get k neighbors
    neighbors = np.empty(k)
    neighbor_arr_idx = 0
    while(neighbors.size < k):
        #there will never be more than 1 minimum because of the distance metric
        neighbor_idx = np.argmin(dist_arr)
        dist_arr[neighbor_idx] = np.inf #dont reuse the same neighbor
        neighbors[neighbor_arr_idx] = neighbor_idx
        neighbor_arr_idx += 1
    return neighbors

#output json file
def out_knn(knn, k, outfile):
    outfile.write("{\n")
    print("printing json: ")
    for i in tqdm(range(len(knn))):
        outfile.write("\t\"" + str(i) + "\": {\n")
        tile = knn[i]
        for j in range(len(tile)):
            outfile.write("\t\t\"" + str(j) + "\": [\n")
            site = tile[j]
            for k in range(len(site)):
                neighbor = site[k]
                outfile.write("\t\t\t")
                outfile.write(str(neighbor))
                outfile.write("\n")
            outfile.write("\t\t]\n")

        outfile.write("\t}\n")
    outfile.write("}")
    
def get_knn(data, tile_size, k):
    #get_knn as a num_tiles * (tile_size * tile_size) * k array
    num_tiles = len(data)
    knn = np.empty((num_tiles, tile_size ** 2, k))
    print("getting tiles: ")
    for i, tile in enumerate(data):
        tile = tile["occupancy_feature"] #only care about occupancy for distance
        tile = tile.squeeze()
        tile = torch.nan_to_num(tile) #get rid of nan's so we can calculate

        #normalize each individual tensor
        tile = torch.nn.functional.normalize(tile,p=2.0,dim=2)

        #tile shape is currently a 5, x size x size array: one tile for each feature
        #we want a size x size x 5 array to get a length 5 list of features per site in a tile
        #so order needs to go from "a, b, c" to "c, b, a"
        tile = torch.transpose(tile, 0, 2)
        tile = torch.transpose(tile, 0, 1)
        tile = tile.reshape((tile_size * tile_size, 5)) #flatten 

        site_index = 0
        print("getting tile", i, "sites:")
        with tqdm(total=len(tile)) as progress_bar:
            for site in tile:
                site_neighbors = get_neighbors(site, site_index, tile, k)
                knn[i][site_index] = site_neighbors
                site_index += 1
                progress_bar.update()
    return knn

file_path = '../'
k = 5
file_path = file_path + '../knn_' + str(k) + '.json'
outfile = open(file_path, "w")
tile_size = 64
data_root = '../bird_data'
data = BirdSpeciesDataset(data_root, tile_size)

dataloader = DataLoader(data, batch_size=1, shuffle=False)

knn = get_knn(dataloader, tile_size, k)
out_knn(knn, k, outfile)


