import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


model_name = 'model_non-graph_0.001.pth'

file_path = f'../saved_predictions/{model_name}/'




plot_path = './plots/'


if not os.path.isdir(plot_path):
	os.makedirs(plot_path)


csv_path = './csv/'

if not os.path.isdir(csv_path):
	os.makedirs(csv_path)


number_batches = 22
batch_size = 32

for idx in range(number_batches):
    occ_true = np.genfromtxt(f'{file_path}/occ_true_{idx}.csv', delimiter=',').astype('float32')
    occ_pred = np.genfromtxt(f'{file_path}/occ_pred_{idx}.csv', delimiter=',').astype('float32')
        
    for tile_idx in range(batch_size):
        occ_true_t = occ_true[tile_idx].reshape(64, -1).T
        occ_pred_t = occ_pred[tile_idx].reshape(64, -1).T
        
        # im_occ_true_t = Image.fromarray(occ_true_t)
        # im_occ_pred_t = Image.fromarray(occ_pred_t)
        
        # im_occ_true_t = im_occ_true_t.convert("L")
        # im_occ_pred_t = im_occ_pred_t.convert("L")

        plt.imshow(occ_true_t, interpolation='none')
        plt.savefig(f'{plot_path}/occ_true_b{idx}_t{tile_idx}.png', bbox_inches='tight', dpi=100) 

        plt.imshow(occ_pred_t, interpolation='none')
        plt.savefig(f'{plot_path}/occ_pred_b{idx}_t{tile_idx}.png', bbox_inches='tight', dpi=100) 

        # im_occ_true_t.save(f'{plot_path}/occ_true_b{idx}_t{tile_idx}.png')
        # im_occ_pred_t.save(f'{plot_path}/occ_pred_b{idx}_t{tile_idx}.png')

        np.savetxt(f'{csv_path}/occ_true_b{idx}_t{tile_idx}.csv', occ_true_t, delimiter=',')
        np.savetxt(f'{csv_path}/occ_pred_b{idx}_t{tile_idx}.csv', occ_pred_t, delimiter=',')






        



