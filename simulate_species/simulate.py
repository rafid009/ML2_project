from ctypes import util
import os
import numpy as np
import torch
import torch.nn.functional as F
from osgeo import gdal
import gc

import utils

d_float = gdal.GDT_Float32 # for continous data
d_int = gdal.GDT_Int16 # for discrete data

def run(occu_features_file, thetas, thetas_conv, total_visits, total_det_feats, det_feat_params, thetas_det, sparsity, tile_size, random_seed, iid_occu):
    
    np.random.seed(random_seed)

    raster = gdal.Open(occu_features_file)
    projection = raster.GetProjection()
    geotransform = raster.GetGeoTransform()
    occu_features_arr = np.array(raster.ReadAsArray())

    total_feats = occu_features_arr.shape[0] # depth
    total_rows = occu_features_arr.shape[1] # height
    total_cols = occu_features_arr.shape[2] # width

    occu_features_arr = occu_features_arr.astype(np.float16)

    if not os.path.isdir('./npy'):
        os.mkdir('./npy')

    utils.save_to_npy(occu_features_arr, './npy/occupancy_features_'+str(tile_size[0])+'.npy', tile_size)
    
    occu_features_normalized_arr = utils.normalize_features(occu_features_arr)
    
    

    del occu_features_arr
    gc.collect()
    
    utils.save_to_tif(occu_features_normalized_arr, "./tif/occupancy_features_normalized.tif", d_float, projection, geotransform)

    true_occupancy_prob = np.zeros(shape=(total_rows, total_cols))
    
#    true_occupancy_prob = thetas[0] * np.zeros(shape=(total_rows, total_cols))

    if (iid_occu==True):
        for feat in range(total_feats):
            true_occupancy_prob += thetas[feat+1] * occu_features_normalized_arr[feat]
        
        true_occupancy_prob += thetas[0]  # add intercept term

    else:
         
        true_occupancy_prob_tensor = torch.from_numpy(true_occupancy_prob)

        for feat in range(total_feats):
            weights = thetas_conv[feat+1] # take kernel for current feature
            weights = weights.view(1, 1, 3, 3) # reshape kernel from (3,3) to (1,1,3,3)
            feature_tensor = torch.tensor(np.expand_dims(occu_features_normalized_arr[feat], 0), dtype=torch.float32 ) # convert current feature to tensor
            convolved = F.conv2d(feature_tensor, weights, padding='same') # apply convolution with padding='same'
            true_occupancy_prob_tensor = torch.add(true_occupancy_prob_tensor, convolved) # add convolved output to running sum
         
        true_occupancy_prob_tensor = torch.add(true_occupancy_prob_tensor, thetas_conv[0]) # add intercept term
        true_occupancy_prob = np.squeeze(true_occupancy_prob_tensor.numpy(), axis=0)
        
    true_occupancy_prob = 1/(1 + np.exp(-true_occupancy_prob))
    
    true_occupancy = true_occupancy_prob.copy()
    
    true_occupancy = np.where(true_occupancy < 0.5, 0, 1)

    nan_indices = np.argwhere(np.isnan(true_occupancy_prob))
    nonnan_indices = np.argwhere(~np.isnan(true_occupancy_prob))

    utils.save_to_npy(true_occupancy, './npy/occupancy_true_'+str(tile_size[0])+'.npy', tile_size)
    utils.save_to_npy(true_occupancy_prob, './npy/occupancy_prob_true_'+str(tile_size[0])+'.npy', tile_size)
    
    utils.save_to_tif(true_occupancy, "./tif/occupancy_true.tif", d_float, projection, geotransform)
    utils.save_to_tif(true_occupancy_prob, "./tif/occupancy_prob_true.tif", d_float, projection, geotransform)

    #print(nan_indices)
    
    del occu_features_normalized_arr, true_occupancy_prob
    gc.collect()

    detection_feature_visits = []

    for visit in range(total_visits):
    
        feature_list = np.zeros(shape=(total_det_feats, total_rows, total_cols))
        for i in range(total_det_feats):
            feature_c = np.random.normal(det_feat_params[i][0], det_feat_params[i][1], total_rows*total_cols).reshape((total_rows,total_cols))
            
            feature_list[i] = feature_c
        feature_list = utils.set_nan(feature_list, nan_indices)
        detection_feature_visits.append(feature_list)
        utils.save_to_npy(feature_list, './npy/detection_features_v'+str(visit+1)+'_'+str(tile_size[0])+'.npy', tile_size)
        utils.save_to_tif(feature_list, "./tif/detection_features_v"+str(visit+1)+".tif", d_float, projection, geotransform)


    detection_arr = np.zeros(shape=(total_visits, total_rows, total_cols), dtype=np.float16)

    for visit in range(total_visits):
        detection_feature_visits[visit] = utils.normalize_features(detection_feature_visits[visit]) # Normalize detection features

        for feat in range(total_det_feats):
            detection_arr[visit] += thetas_det[feat+1] * detection_feature_visits[visit][feat]
       
        detection_arr[visit] += thetas_det[0]  # add intercept term
        detection_arr[visit] = 1/(1 + np.exp(-detection_arr[visit]))
        detection_arr[visit] = detection_arr[visit] * true_occupancy
        detection_arr[visit] = np.where(detection_arr[visit] < 0.5, 0, 1)


    utils.save_to_npy(detection_arr, './npy/detection_'+str(tile_size[0])+'.npy', tile_size)
    utils.save_to_tif(detection_arr, "./tif/detection.tif", d_int, projection, geotransform)

    np.random.shuffle(nonnan_indices)
    
    del detection_feature_visits, true_occupancy
    gc.collect()

    removal_site_no = int(sparsity * (nonnan_indices.shape[0]))
    removal_indices = nonnan_indices[:removal_site_no]

    utils.set_nan(detection_arr, removal_indices)
    
    utils.save_to_npy(detection_arr, './npy/detection_'+str(int(sparsity*100))+'_'+str(tile_size[0])+'.npy', tile_size)
    utils.save_to_tif(detection_arr, './tif/detection_'+str(int(sparsity*100))+'.tif', d_int, projection, geotransform)


