# library imports
import torch as th
import numpy as np
from pytorch3d.loss import chamfer_distance as cd
#import umap
#import matplotlib.pyplot as plt
#import seaborn as sns
#import pandas as pd
#from sklearn.preprocessing import StandardScaler
#from sklearn.neighbors import NearestNeighbors

# loading in the data
conv_hidden_output_baseline_subset = np.load('./Data/partseg_full_data_conv9/conv9_hidden_output_full_data_predictions.npy', allow_pickle = True).tolist()


# chamfer distance matrix calculation using Pytorch3D
chamfer_dist_matrix_baseline_subset_numpy = np.asarray([[list(cd(th.from_numpy(np.resize(p1, (1,p1.shape[0],256))).cuda(), th.from_numpy(np.resize(p2, (1,p2.shape[0],256))).cuda()))[0].cpu() for p2 in conv_hidden_output_baseline_subset] for p1 in conv_hidden_output_baseline_subset])

#chamfer_dist_matrix_baseline_subset_numpy = np.asarray(chamfer_dist_matrix_baseline_subset)
np.save('./Results/conv9_chamfer_dist_matrix_full_data_predictions_numpy_cuda.npy', chamfer_dist_matrix_baseline_subset_numpy)
print(chamfer_dist_matrix_baseline_subset_numpy.shape)


#chamfer_dist_matrix_baseline_subset_numpy = np.asarray(chamfer_dist_matrix_baseline_subset)[:,:,0]
#print(chamfer_dist_matrix_baseline_subset_numpy.shape)
#np.save('./Results/conv9_chamfer_dist_matrix_full_data_predictions_numpy_cuda.npy', chamfer_dist_matrix_baseline_subset_numpy)
