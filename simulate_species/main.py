import torch
import simulate

# Please keep occupancy feature tif file in ./tif
occu_features_file = "./tif/occupancy_features.tif" # Occupancy Feature Raster tif file

# Weights for pixel-wise model (if used) which creates species in iid. manner
thetas = [	
            -1.8, # Intercept
            0.5, # Elevation
            -0.5, # TCB - Brightness
            1.25,  # TCG - Greenness 
            1.0, # TCW - Wetness
            -0.3 # TCA - Angle
        ]

# Weights for convolutional kernel model (if used) which creates species in non iid. manner
thetas_conv = [
                torch.tensor(-2.55),  #Intercept

                torch.tensor([[0.05, 0.05, 0.05],    # Elevation
                              [0.05, 0.5, 0.05],
                              [0.05, 0.05, 0.05]]),
                torch.tensor([[-0.05, -0.05, -0.05],   # TCB - Brightness
                              [-0.05, -0.5, -0.05],
                              [-0.05, -0.05, -0.05]]),
                torch.tensor([[0.125, 0.125, 0.125],   # TCG - Greenness 
                              [0.125, 1.25, 0.125],
                              [0.125, 0.125, 0.125]]),  
                torch.tensor([[0.01, 0.01, 0.01],    # TCW - Wetness
                              [0.01, 1.0, 0.01],
                              [0.01, 0.1, 0.01]]),
                torch.tensor([[-0.03, -0.03, -0.03],  # TCA - Angle
                              [-0.03, -0.3, -0.03],
                              [-0.03, -0.03, -0.03]]),
]


total_visits = 5 # Total number of visits per site

total_det_feats = 3 # Number of detection features

det_feat_params = [ 
                    [ 1, 0.3], # [mu, sigma] for W1
                    [ 2, 0.2], # [mu, sigma] for W2
                    [ 3, 0.1]  # [mu, sigma] for W3
]

# Arbitrary weights, intercept is varied
thetas_det = [	
            -0.3, # Intercept  
            0.1, # W1
            0.2, # W2
            0.3,  # W3 
        ]

sparsity = 0.6

tile_size = (64,64)

random_seed = 1

iid_occu = True
iid_occu = False


simulate.run(
            occu_features_file=occu_features_file, 
            thetas=thetas,
            thetas_conv=thetas_conv, 
            total_visits=total_visits,
            total_det_feats=total_det_feats,
            det_feat_params=det_feat_params,
            thetas_det=thetas_det,
            sparsity=sparsity,
            tile_size=tile_size, 
            random_seed=random_seed,
            iid_occu=iid_occu,
            )


