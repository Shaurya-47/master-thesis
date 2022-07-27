# library imports
import torch as th
import numpy as np
import umap
#import matplotlib.pyplot as plt
#import seaborn as sns
#import pandas as pd

from scipy.sparse import csr_matrix, csgraph
from umap.umap_ import compute_membership_strengths, smooth_knn_dist, make_epochs_per_sample, simplicial_set_embedding, find_ab_params
import scipy
#from sklearn.preprocessing import StandardScaler
#from sklearn.neighbors import NearestNeighbors

# loading the 2D embedding and UMAP graph
graph_sparse = scipy.sparse.load_npz('./Results/conv9_graph_full_data_hp_0.5_300_rs_1.npz')
#embedding = np.load('./Data/partseg_full_data_conv9/conv9_UMAP_embedding_hp_0.5_300_rs_1.npy')

# importing data - default numpy array format

airplane_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/airplane_test_subset_part_labels_big_dataset.pt')
bag_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/bag_test_subset_part_labels_big_dataset.pt')
cap_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/cap_test_subset_part_labels_big_dataset.pt')
car_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/car_test_subset_part_labels_big_dataset.pt')
chair_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/chair_test_subset_part_labels_big_dataset.pt')
earphone_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/earphone_test_subset_part_labels_big_dataset.pt')
guitar_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/guitar_test_subset_part_labels_big_dataset.pt')
knife_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/knife_test_subset_part_labels_big_dataset.pt')
lamp_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/lamp_test_subset_part_labels_big_dataset.pt')
laptop_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/laptop_test_subset_part_labels_big_dataset.pt')
motorbike_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/motorbike_test_subset_part_labels_big_dataset.pt')
mug_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/mug_test_subset_part_labels_big_dataset.pt')
pistol_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/pistol_test_subset_part_labels_big_dataset.pt')
rocket_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/rocket_test_subset_part_labels_big_dataset.pt')
skateboard_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/skateboard_test_subset_part_labels_big_dataset.pt')
table_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/table_test_subset_part_labels_big_dataset.pt')

airplane_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/airplane_test_subset_predictions_big_dataset.pt')
bag_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/bag_test_subset_predictions_big_dataset.pt')
cap_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/cap_test_subset_predictions_big_dataset.pt')
car_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/car_test_subset_predictions_big_dataset.pt')
chair_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/chair_test_subset_predictions_big_dataset.pt')
earphone_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/earphone_test_subset_predictions_big_dataset.pt')
guitar_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/guitar_test_subset_predictions_big_dataset.pt')
knife_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/knife_test_subset_predictions_big_dataset.pt')
lamp_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/lamp_test_subset_predictions_big_dataset.pt')
laptop_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/laptop_test_subset_predictions_big_dataset.pt')
motorbike_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/motorbike_test_subset_predictions_big_dataset.pt')
mug_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/mug_test_subset_predictions_big_dataset.pt')
pistol_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/pistol_test_subset_predictions_big_dataset.pt')
rocket_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/rocket_test_subset_predictions_big_dataset.pt')
skateboard_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/skateboard_test_subset_predictions_big_dataset.pt')
table_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/table_test_subset_predictions_big_dataset.pt')


# concatenating predictions subset and part labels subset

part_labels_subset = np.vstack((airplane_test_subset_part_labels,
                                bag_test_subset_part_labels,
                                cap_test_subset_part_labels,
                                car_test_subset_part_labels,
                                chair_test_subset_part_labels,
                                earphone_test_subset_part_labels,
                                guitar_test_subset_part_labels,
                                knife_test_subset_part_labels,
                                lamp_test_subset_part_labels,
                                laptop_test_subset_part_labels,
                                motorbike_test_subset_part_labels,
                                mug_test_subset_part_labels,
                                pistol_test_subset_part_labels,
                                rocket_test_subset_part_labels,
                                skateboard_test_subset_part_labels,
                                table_test_subset_part_labels
                                ))

predictions_subset = np.vstack((airplane_test_subset_predictions,
                                bag_test_subset_predictions,
                                cap_test_subset_predictions,
                                car_test_subset_predictions,
                                chair_test_subset_predictions,
                                earphone_test_subset_predictions,
                                guitar_test_subset_predictions,
                                knife_test_subset_predictions,
                                lamp_test_subset_predictions,
                                laptop_test_subset_predictions,
                                motorbike_test_subset_predictions,
                                mug_test_subset_predictions,
                                pistol_test_subset_predictions,
                                rocket_test_subset_predictions,
                                skateboard_test_subset_predictions,
                                table_test_subset_predictions
                                ))

predictions_subset = th.from_numpy(predictions_subset)
predictions_subset = predictions_subset.permute(0, 2, 1).contiguous()
predictions_subset = predictions_subset.max(dim=2)[1]
predictions_subset = predictions_subset.detach().cpu().numpy()
predictions_subset = np.resize(predictions_subset, (2002944,1)).flatten()
np.unique(predictions_subset)


part_labels_subset = np.resize(part_labels_subset, (2002944,1)).flatten()
np.unique(part_labels_subset)

print(np.unique(predictions_subset).shape) # 49/50 as a chair part and a motorcycle part are never predicted
print(np.unique(part_labels_subset).shape) # 49/50 as one label does not exist in data


########################## PREDICTIONS ########################################

# creating an index list for the start of each example
examples = []
for i in range(len(predictions_subset)):
    if i % 2048 == 0:
        examples.append(i)
examples_forward = examples[1:]

# creating the range sets per example 
example_ranges = []
for j,k in zip(examples, examples_forward):
    example_ranges.append(range(j,k))

# appending the last index to the list
example_ranges.append(range(len(predictions_subset)-2048, len(predictions_subset)))

# check
print(example_ranges[1][0],example_ranges[1][-1])
print(len(example_ranges[1]))
print(example_ranges)


# edited version where the part and intersection Booleans are in the shape 
# of the Adjacency Matrix
part_indices_outer_list = []
part_outer_list = []

# iterating over all example ranges
for ran in example_ranges:
    part_indices_list = []
    part_list = []
    
    # iterating over all the parts in the range of an example and getting 
    # the indices of the rows containing the part (via prediction/label)
    for part in np.unique(predictions_subset[ran[0]:ran[-1]+1]):

        part_indices = np.array(predictions_subset == part)
        part_indices[0:ran[0]] = False
        part_indices[ran[-1]+1:len(predictions_subset)] = False
        part_indices_list.append(part_indices)
  
        part_list.append(part)
        
    part_indices_outer_list.append(part_indices_list)
    part_outer_list.append(part_list)
    

part_indices_outer_list_flattened = [part for sublist in part_indices_outer_list for part in sublist]
part_outer_list_flattened = [part for sublist in part_outer_list for part in sublist]


# FINAL NESTED FOR LOOP

mean_weights_list = []
for i in range(len(part_outer_list_flattened)):
    inner_list = []
    graph_sparse_0 = graph_sparse[part_indices_outer_list_flattened[i].nonzero()[0],:]
    for j in range(len(part_outer_list_flattened)):
        if i != j:
            graph_sparse_1 = graph_sparse_0.transpose()[part_indices_outer_list_flattened[j].nonzero()[0],:]
            sum_weights = graph_sparse_1.sum()
            mean_weights = sum_weights/graph_sparse_1.count_nonzero()
            inner_list.append(mean_weights) 
        else:
            # same index intersection not appended 
            inner_list.append(0) # to append 0 at same example part positions
    mean_weights_list.append(inner_list)


mean_weights_list_numpy = np.array(mean_weights_list)
mean_weights_list_numpy = np.nan_to_num(mean_weights_list_numpy)

graph_new = csr_matrix(mean_weights_list_numpy)

scipy.sparse.save_npz('./Results/conv9_protocol_two_graph_300_0.5_rs_1.npz', graph_new)