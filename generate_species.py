import numpy as np
from osgeo import gdal
import gc

def split_tiles(arr, tile_size):
    r_dim = 0
    c_dim = 1
    tiles = None 
   
    if (len(arr.shape)==3):
        r_dim = 1
        c_dim = 2
    
    rows = arr.shape[r_dim]
    cols = arr.shape[c_dim]

    row_tile_no = rows//tile_size[0]
    col_tile_no = cols//tile_size[1]

    rows = rows - (rows - (row_tile_no * tile_size[0]))
    cols = cols - (cols - (col_tile_no * tile_size[1]))

    if (len(arr.shape)==3):
        tiles = np.zeros(shape=(row_tile_no*col_tile_no, arr.shape[0], tile_size[0], tile_size[1]))
        arr = arr[:, :rows, :cols]
    else:
        tiles = np.zeros(shape=(row_tile_no*col_tile_no, tile_size[0], tile_size[1]))
        arr = arr[:rows, :cols]
    
   
    row_cuts = np.array(np.array_split(arr, row_tile_no, r_dim))

    #print(arr)
    #print(arr, '\n')
    k = 0
    for i, row_cut in enumerate(row_cuts):
      #  print(row_cut)
       # print('\n')
        col_cuts = np.array(np.array_split(row_cut, col_tile_no, c_dim))
        for j, col_cut in enumerate(col_cuts):
           # print(col_cut)
            tiles[i*col_tile_no + j] = col_cut
        k += 1
            
    return tiles

occu_feat_names = ["elevation", "TCB", "TCG", "TCW", "TCA"]

raster = gdal.Open("occupancy_features.tif")

imarray = np.array(raster.ReadAsArray())

#print(np.count_nonzero(~np.isnan(imarray[1])))

#print(imarray.shape)

total_feats = imarray.shape[0] # depth
total_rows = imarray.shape[1] # height
total_cols = imarray.shape[2] # width



imarray = imarray.astype(np.float32)


# Write Occupancy Features to .npy file
np.save('./npy_files/T64/occupancy_features_64.npy', split_tiles(imarray, (64, 64))) # 64 x 64 tiles
np.save('./npy_files/T128/occupancy_features_128.npy', split_tiles(imarray, (128, 128))) # 128 x 128 tiles


#print(imarray[0, 2000:2010, 2000:2010])
#print(imarray[1, 2000:2010, 2000:2010])

# Normalize features
imarray_n = imarray.copy()





#print(nan_indices[:10])

#print(np.nanmax(imarray[0]))
#print(imarray[0].shape)

for feat in range(total_feats):
    min_val = np.nanmin(imarray[feat])
    max_val = np.nanmax(imarray[feat])
 
    imarray_n[feat] = (imarray[feat] - min_val) / (max_val - min_val)
    


#print(imarray_n[0, 2000:2010, 2000:2010])
#print(imarray_n[1, 2000:2010, 2000:2010])


# create the output image

driver = gdal.GetDriverByName("GTiff")
outDs = driver.Create("occupancy_features_normalized.tif", total_cols, total_rows, total_feats, gdal.GDT_Float32)

# georeferencing parameters 
outDs.SetProjection(raster.GetProjection())
outDs.SetGeoTransform(raster.GetGeoTransform())

# Write raster data sets
for i in range(total_feats):
    outBand = outDs.GetRasterBand(i + 1)
    outBand.WriteArray(imarray_n[i])

# Close raster file
outDs = None

del outDs, outBand


true_occupancy = np.zeros(shape=(total_rows, total_cols))
true_occupancy_prob = np.zeros(shape=(total_rows, total_cols))

#true_occupancy = imarray[0].copy()
#true_occupancy_prob = imarray[0].copy()

thetas = [	
            -1.8, # Intercept
            0.5, # Elevation
            -0.5, # TCB - Brightness
            1.25,  # TCG - Greenness 
            1.0, # TCW - Wetness
            -0.3 # TCA - Angle
        ]

for row in range(total_rows):
    for col in range(total_cols):
        
        z = thetas[0] # intercept
        for feat in range(total_feats):
           z += (thetas[feat+1] * imarray_n[feat, row, col])
           
        true_occupancy_prob[row, col] = 1/(1 + np.exp(-z)) # sigmoid
        
        if(not np.isnan(true_occupancy_prob[row, col])):
            true_occupancy[row, col] = int(true_occupancy_prob[row, col]>=0.5)



#true_occupancy = np.where(np.isnan(true_occupancy_prob) , np.nan, true_occupancy)


nan_indices = np.argwhere(np.isnan(true_occupancy_prob))
nonnan_indices = np.argwhere(~np.isnan(true_occupancy_prob))
print(len(nan_indices))
print(len(nonnan_indices))

for idx in nan_indices:
    true_occupancy[idx[0],idx[1]] = np.nan
   # print(idx[0], idx[1])



driver = gdal.GetDriverByName("GTiff")
outDs = driver.Create("occupancy_true.tif", total_cols, total_rows, 1, gdal.GDT_Float32)



#true_occupancy_prob[nan_indices] =  float("NaN")

# georeferencing parameters 
outDs.SetProjection(raster.GetProjection())
outDs.SetGeoTransform(raster.GetGeoTransform())

outBand = outDs.GetRasterBand(1)
outBand.WriteArray(true_occupancy)

# Close raster file
outDs = None

del outDs, outBand

# Write Occupancy to .npy file
np.save('./npy_files/T64/occupancy_true_64.npy', split_tiles(true_occupancy, (64, 64))) # 64 x 64 tiles
np.save('./npy_files/T128/occupancy_true_128.npy', split_tiles(true_occupancy, (128, 128))) # 128 x 128 tiles

driver = gdal.GetDriverByName("GTiff")
outDs = driver.Create("occupancy_prob_true.tif", total_cols, total_rows, 1, gdal.GDT_Float32)

# georeferencing parameters 
outDs.SetProjection(raster.GetProjection())
outDs.SetGeoTransform(raster.GetGeoTransform())

outBand = outDs.GetRasterBand(1)
outBand.WriteArray(true_occupancy_prob)

# Close raster file
outDs = None

del outDs, outBand

# Write Occupancy Probs to .npy file
np.save('./npy_files/T64/occupancy_true_prob_64.npy', split_tiles(true_occupancy_prob, (64, 64))) # 64 x 64 tiles
np.save('./npy_files/T128/occupancy_true_prob_128.npy', split_tiles(true_occupancy_prob, (128, 128))) # 128 x 128 tiles

# Create detection features

# Constant detection features over all sites but not across visits.
# Each detection feature is simulated to have Normal distributions over visits

total_visits = 5 # Total number of visits per site
total_det_feats = 3

det_feat_params = [ 
                    [ 1, 0.3], # [mu, sigma] for W1
                    [ 2, 0.2], # [mu, sigma] for W2
                    [ 3, 0.1]  # [mu, sigma] for W3
]

np.random.seed(1)

#print(det_feat_vals)

# Write detection features to .tif files

detection_feature_tensors = []

for visit in range(total_visits):
    driver = gdal.GetDriverByName("GTiff")
    outDs = driver.Create("./detection_features/detection_features_v"+str(visit+1)+".tif", total_cols, total_rows, total_det_feats, gdal.GDT_Float32)

    # georeferencing parameters 
    outDs.SetProjection(raster.GetProjection())
    outDs.SetGeoTransform(raster.GetGeoTransform())

    feature_tensors = np.zeros(shape=(total_det_feats, total_rows, total_cols))
    # Write raster data sets
    for i in range(total_det_feats):
        feature_tensor = np.random.normal(det_feat_params[i][0], det_feat_params[i][1], total_rows*total_cols).reshape((total_rows,total_cols))
        for idx in nan_indices:
            feature_tensor[idx[0],idx[1]] = np.nan

       
        
        feature_tensors[i] = feature_tensor
        outBand = outDs.GetRasterBand(i + 1)
        outBand.WriteArray(feature_tensor)

     # Write Detection Features to .npy file
    np.save('./npy_files/T64/detection_features_v'+str(visit+1)+'_64.npy', split_tiles(feature_tensors, (64, 64))) # 64 x 64 tiles
    np.save('./npy_files/T128/detection_features_v'+str(visit+1)+'128.npy', split_tiles(feature_tensors, (128, 128))) # 128 x 128 tiles
        
    print(feature_tensors.shape)
    detection_feature_tensors.append(feature_tensors)
    

    # Close raster file
    outDs = None

    del outDs, outBand



# Normalize detection features
for visit in range(total_visits):
    for feat in range(total_det_feats):
        min_val = np.nanmin(detection_feature_tensors[visit][feat])
        max_val = np.nanmax(detection_feature_tensors[visit][feat])
        detection_feature_tensors[visit][feat] = (detection_feature_tensors[visit][feat] - min_val) / (max_val - min_val)
    
print("Reached detection layer")

# Write detection layer to tif. file

detection_layer = np.zeros(shape=(total_visits, total_rows, total_cols), dtype=np.float16)

# Arbitrary weights, intercept is varied
thetas_det = [	
            -0.3, # Intercept  # -0.25 (option)
            0.1, # W1
            0.2, # W2
            0.3,  # W3 
        ]

for row in range(total_rows):
    for col in range(total_cols):
        
        for visit in range(total_visits):
    
            z = thetas_det[0] # intercept
            for feat in range(total_det_feats):
               z += (thetas_det[feat+1] * detection_feature_tensors[visit][feat, row, col])
            
            det_prob = 1/(1 + np.exp(-z)) # sigmoid

            detection_layer[visit, row, col] = int(det_prob> 0.5) * true_occupancy[row,col]

for visit in range(total_visits):
    for idx in nan_indices:
        detection_layer[visit, idx[0],idx[1]] = np.nan




del imarray, imarray_n, true_occupancy, true_occupancy_prob, detection_feature_tensors
gc.collect()


driver = gdal.GetDriverByName("GTiff")
outDs = driver.Create("detection.tif", total_cols, total_rows, total_visits, gdal.GDT_Int16)

# georeferencing parameters 
outDs.SetProjection(raster.GetProjection())
outDs.SetGeoTransform(raster.GetGeoTransform())


for i in range(total_visits):
    outBand = outDs.GetRasterBand(i + 1)
    outBand.WriteArray(detection_layer[i])

#outBand = outDs.GetRasterBand(1)
#outBand.WriteArray(detection_layer)

# Close raster file
outDs = None

del outDs, outBand

# Write Detection to .npy file
np.save('./npy_files/T64/detection_64.npy', split_tiles(detection_layer, (64, 64))) # 64 x 64 tiles
np.save('./npy_files/T128/detection_128.npy', split_tiles(detection_layer, (128, 128))) # 128 x 128 tiles


print("Reached detection layer 80")
# Introduce sparsity

np.random.shuffle(nonnan_indices)

# 80 $ Sparsity - 20% sites labeled

sparsity = 0.8


removal_site_no = int(0.8 * (nonnan_indices.shape[0]))
removal_indices = nonnan_indices[:removal_site_no]

detection_layer_80 = detection_layer.copy()
detection_layer_60 = detection_layer.copy()

del detection_layer
gc.collect()

for visit in range(total_visits):
    for idx in removal_indices:
        detection_layer_80[visit, idx[0],idx[1]] = np.nan

driver = gdal.GetDriverByName("GTiff")
outDs = driver.Create("detection_80.tif", total_cols, total_rows, total_visits, gdal.GDT_Int16)

# georeferencing parameters 
outDs.SetProjection(raster.GetProjection())
outDs.SetGeoTransform(raster.GetGeoTransform())

for i in range(total_visits):
    outBand = outDs.GetRasterBand(i + 1)
    outBand.WriteArray(detection_layer_80[i])


# Close raster file
outDs = None

del outDs, outBand

# Write Detection (80% sparsity) to .npy file
np.save('./npy_files/T64/detection_80_64.npy', split_tiles(detection_layer_80, (64, 64))) # 64 x 64 tiles
np.save('./npy_files/T128/detection_80_128.npy', split_tiles(detection_layer_80, (128, 128))) # 128 x 128 tiles


del detection_layer_80
gc.collect()


# 60 $ Sparsity - 40% sites labeled

sparsity = 0.6

removal_site_no = int(0.6 * (nonnan_indices.shape[0]))
removal_indices = nonnan_indices[:removal_site_no]




for visit in range(total_visits):
    for idx in removal_indices:
        detection_layer_60[visit, idx[0],idx[1]] = np.nan


driver = gdal.GetDriverByName("GTiff")
outDs = driver.Create("detection_60.tif", total_cols, total_rows, total_visits, gdal.GDT_Int16)

# georeferencing parameters 
outDs.SetProjection(raster.GetProjection())
outDs.SetGeoTransform(raster.GetGeoTransform())


 
for i in range(total_visits):
    outBand = outDs.GetRasterBand(i + 1)
    outBand.WriteArray(detection_layer_60[i])


# Close raster file
outDs = None

del outDs, outBand

# Write Detection (60% sparsity) to .npy file
np.save('./npy_files/T64/detection_60_64.npy', split_tiles(detection_layer_60, (64, 64))) # 64 x 64 tiles
np.save('./npy_files/T128/detection_60_128.npy', split_tiles(detection_layer_60, (128, 128))) # 128 x 128 tiles