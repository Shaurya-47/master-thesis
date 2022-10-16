# library imports
import torch as th
import numpy as np
import umap
import matplotlib.pyplot as plt
import pandas as pd
from pytorch3d.loss import chamfer_distance as cd

# importing data - default numpy array format
conv8_output = th.load('semseg_test_conv8_hidden_output_2048_100.pt')
labels = th.load('semseg_test_labels_2048_100.pt')
predictions = th.load('semseg_test_predictions_2048_100.pt')

# dropping the batch size dimension of the tensors
conv8_output = np.array(conv8_output)
conv8_output = np.moveaxis(conv8_output, 2, 1)
conv8_output = np.resize(conv8_output, (204800,256))

# part labels
labels = labels.flatten()
# predictions
predictions = predictions.permute(0, 2, 1).contiguous()
predictions = predictions.max(dim=2)[1]
predictions = predictions.detach().cpu().numpy()
predictions = np.resize(predictions, (204800,1)).flatten()

# part masks - match number of parts with color scheme
indices_part_0 = np.array(predictions == 0)
indices_part_1 = np.array(predictions == 1)
indices_part_2 = np.array(predictions == 2)
indices_part_3 = np.array(predictions == 3)
indices_part_4 = np.array(predictions == 4)
indices_part_5 = np.array(predictions == 5)
indices_part_6 = np.array(predictions == 6)
indices_part_7 = np.array(predictions == 7)
indices_part_8 = np.array(predictions == 8)
indices_part_9 = np.array(predictions == 9)
indices_part_10 = np.array(predictions == 10)
indices_part_11 = np.array(predictions == 11)
indices_part_12 = np.array(predictions == 12)


# applying part mask to get subsets for all examples

conv_8_hidden_output_part_0 = []
conv_8_hidden_output_part_1 = []
conv_8_hidden_output_part_2 = []
conv_8_hidden_output_part_3 = []
conv_8_hidden_output_part_4 = []
conv_8_hidden_output_part_5 = []
conv_8_hidden_output_part_6 = []
conv_8_hidden_output_part_7 = []
conv_8_hidden_output_part_8 = []
conv_8_hidden_output_part_9 = []
conv_8_hidden_output_part_10 = []
conv_8_hidden_output_part_11 = []
conv_8_hidden_output_part_12 = []


for i in range(predictions.shape[0]):
    
    # subsetting
    inner_result_0 = conv8_output[i][indices_part_0[i]]
    inner_result_1 = conv8_output[i][indices_part_1[i]]
    inner_result_2 = conv8_output[i][indices_part_2[i]]
    inner_result_3 = conv8_output[i][indices_part_3[i]]
    inner_result_4 = conv8_output[i][indices_part_4[i]]
    inner_result_5 = conv8_output[i][indices_part_5[i]]
    inner_result_6 = conv8_output[i][indices_part_6[i]]
    inner_result_7 = conv8_output[i][indices_part_7[i]]
    inner_result_8 = conv8_output[i][indices_part_8[i]]
    inner_result_9 = conv8_output[i][indices_part_9[i]]
    inner_result_10 = conv8_output[i][indices_part_10[i]]
    inner_result_11 = conv8_output[i][indices_part_11[i]]
    inner_result_12 = conv8_output[i][indices_part_12[i]]
    
        
    # appending
    conv_8_hidden_output_part_0.append(inner_result_0)
    conv_8_hidden_output_part_1.append(inner_result_1)
    conv_8_hidden_output_part_2.append(inner_result_2)
    conv_8_hidden_output_part_3.append(inner_result_3)
    conv_8_hidden_output_part_4.append(inner_result_4)
    conv_8_hidden_output_part_5.append(inner_result_5)
    conv_8_hidden_output_part_6.append(inner_result_6)
    conv_8_hidden_output_part_7.append(inner_result_7)
    conv_8_hidden_output_part_8.append(inner_result_8)
    conv_8_hidden_output_part_9.append(inner_result_9)
    conv_8_hidden_output_part_10.append(inner_result_10)
    conv_8_hidden_output_part_11.append(inner_result_11)
    conv_8_hidden_output_part_12.append(inner_result_12)

# checking part counts
conv8_arrays =                       np.vstack((conv_8_hidden_output_part_0,
                                                conv_8_hidden_output_part_1,
                                                conv_8_hidden_output_part_2,
                                                conv_8_hidden_output_part_3,
                                                conv_8_hidden_output_part_4,
                                                conv_8_hidden_output_part_6,
                                                conv_8_hidden_output_part_7,
                                                conv_8_hidden_output_part_8,
                                                conv_8_hidden_output_part_10,
                                                conv_8_hidden_output_part_11,
                                                conv_8_hidden_output_part_12))
conv8_hidden_output_baseline_subset = []
part_counts = []
for arr in conv8_arrays:
    inner_boolean = []
    for k in range(0,len(arr)):
        if arr[k].any() == False:
            inner_boolean.append(False)
        else:
            inner_boolean.append(True)
    conv8_hidden_output_baseline_subset.append(arr[np.array(inner_boolean)])
    part_counts.append(np.sum(np.array(inner_boolean)))
# updated part count resizing
conv8_hidden_output_baseline_subset = np.array(conv8_hidden_output_baseline_subset)
conv8_hidden_output_baseline_subset = np.resize(conv8_hidden_output_baseline_subset,
                                                (np.array(part_counts).sum(),1)).flatten()
    
# stacking lists
#parts 5 and 9 do not exist in the data - removing 
conv8_hidden_output_baseline_subset = np.vstack((conv_8_hidden_output_part_0,
                                                conv_8_hidden_output_part_1,
                                                conv_8_hidden_output_part_2,
                                                conv_8_hidden_output_part_3,
                                                conv_8_hidden_output_part_4,
                                                conv_8_hidden_output_part_6,
                                                conv_8_hidden_output_part_7,
                                                conv_8_hidden_output_part_8,
                                                conv_8_hidden_output_part_10,
                                                conv_8_hidden_output_part_11,
                                                conv_8_hidden_output_part_12))

conv8_hidden_output_baseline_subset = np.resize(conv8_hidden_output_baseline_subset, (1100,1)).flatten()
np.save('semseg_conv8_hidden_output_chamfer_input_2048_100_predictions.npy', conv8_hidden_output_baseline_subset)

# Chamfer Distance using PyTorch3d
# lower the chamfer distance, more the similar the point clouds 
chamfer_dist_matrix_baseline_subset = np.asarray([[cd(th.from_numpy(np.resize(p1, (1,p1.shape[0],256))), th.from_numpy(np.resize(p2, (1,p2.shape[0],256)))) for p2 in conv8_hidden_output_baseline_subset] for p1 in conv8_hidden_output_baseline_subset])
np.save('semseg_conv8_chamfer_dist_matrix_baseline_subset_predictions_2048_100_predictions.npy', chamfer_dist_matrix_baseline_subset)
cdm_baseline_subset = np.vectorize(lambda x: x.item())(chamfer_dist_matrix_baseline_subset)
np.mean(cdm_baseline_subset)

# drop nan rows
cdm_baseline_subset = cdm_baseline_subset.flatten()
cdm_baseline_subset = cdm_baseline_subset[~np.isnan(cdm_baseline_subset)]
cdm_baseline_subset = np.reshape(cdm_baseline_subset, (471,471))

# mapping pre-requisites

# colors =   {"ceiling": "#9E9E9E",#grey
#             "floor": "#795548",#brown
#             "wall": "#FF5722",#d orange
#             "beam": "#FFC107",#amber(dark yellow)
#             "column": "#FFEE58",#light yellow
#             "window": "#CDDC39",#lime
#             "door": "#4CAF50",#green
#             "table": "#009688",#table
#             "chair": "#00BCD4",#cyan-light blue
#             "sofa": "#2196F3",#blue
#             "bookcase": "#3F51B5",#indigo(dark blue)
#             "board": "#9C27B0",#purple
#             "clutter": "#E91E63",#pink
#             }

colormap = []
# ceiling
colormap.append(['#9E9E9E'] * 92)
# floor
colormap.append(['#795548'] * 97)
# wall
colormap.append(['#FF5722'] * 69)
# beam
colormap.append(['#FFC107'] * 35)
# column
colormap.append(['#FFEE58'] * 14)
# window
#colormap.append(['#CDDC39'] * )
# door
colormap.append(['#4CAF50'] * 42)
# table
colormap.append(['#009688'] * 14)
# chair
colormap.append(['#00BCD4'] * 14)
# sofa
#colormap.append(['#2196F3'] * )
# bookcase
colormap.append(['#3F51B5'] * 11)
# board
colormap.append(['#9C27B0'] * 6)
# clutter
colormap.append(['#E91E63'] * 77)

# overall = 471 (matches)

colormap = [item for sublist in colormap for item in sublist]
colormap = pd.Series(colormap)
colormap.unique()

# applying UMAP
reducer = umap.UMAP(min_dist = 1, n_neighbors = 200, metric = 'precomputed',
                    random_state = 1)

# applying UMAP to reduce the dimensionality to 2
embedding = reducer.fit_transform(cdm_baseline_subset)
embedding.shape       

# plotting
plt.figure(figsize=(8,6), dpi=500)
plt.scatter(embedding[:,0], embedding[:,1], s=7, marker = 'v', c=colormap)
plt.title('PT-1 projection (using predictions) \n on 100 samples (cloud size 1024)', fontsize=24)
