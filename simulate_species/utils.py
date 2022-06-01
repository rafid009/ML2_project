import numpy as np
from osgeo import gdal
import gc

def set_nan (arr, indices):
    if (len(arr.shape)==3): # check for multi channel
        for c in range(arr.shape[0]):
            for idx in indices:
                arr[c, idx[0],idx[1]] = np.nan
    else:
        for idx in indices:
            arr[idx[0],idx[1]] = np.nan
    return arr

def save_to_npy(arr, file, tile_size):

	np.save(file, split_tiles(arr, tile_size)) # 64 x 64 tiles
	

def save_to_tif(arr, file, dtype, projection, geotransform):
	
    driver = gdal.GetDriverByName("GTiff")
    outDs = None
    
    outBand = None
    if (len(arr.shape)==3): # check for multi channel

        outDs = driver.Create(file, arr.shape[2], arr.shape[1], arr.shape[0], dtype)
        # georeferencing parameters 
        outDs.SetProjection(projection)
        outDs.SetGeoTransform(geotransform)

        for i in range(arr.shape[0]):
            outBand = outDs.GetRasterBand(i + 1)
            outBand.WriteArray(arr[i])

    else:
        outDs = driver.Create(file, arr.shape[1], arr.shape[0], 1, dtype)
        # georeferencing parameters 
        outDs.SetProjection(projection)
        outDs.SetGeoTransform(geotransform)

        outBand = outDs.GetRasterBand(1)
        outBand.WriteArray(arr)

    # Close raster file
    outDs = None
    del outDs, outBand

	

def normalize_features(arr):
    arr_normalized = arr.copy()

    for feat in range(arr.shape[0]):
        min_val = np.nanmin(arr[feat])
        max_val = np.nanmax(arr[feat])
    
        arr_normalized[feat] = (arr[feat] - min_val) / (max_val - min_val) # Vectorized elementwise min-max normalization

    return arr_normalized


def split_tiles(arr, tile_size):
    r_dim = 0
    c_dim = 1
    tiles = None 
   
   
    if (len(arr.shape)==3): # check for multi channel
        r_dim = 1
        c_dim = 2
    
    rows = arr.shape[r_dim]
    cols = arr.shape[c_dim]

    row_tile_no = rows//tile_size[0]
    col_tile_no = cols//tile_size[1]

    rows = rows - (rows - (row_tile_no * tile_size[0]))
    cols = cols - (cols - (col_tile_no * tile_size[1]))

    if (len(arr.shape)==3): # check for multi channel
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