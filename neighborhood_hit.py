# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 17:27:05 2022

@author: Shaurya
"""

import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(1)

blobs = datasets.make_blobs(n_samples=400, random_state=1, centers=3)

labels = blobs[1]
data = blobs[0]

# plot data
plt.scatter(data[:,0], data[:,1], s=7, marker = 'v')


# defining a fucntion for NH
def neighborhood_hit(data, labels, n_neighbors):
    
    # getting the KNN indices for each point in the dataset
    neighbors = KNeighborsClassifier(n_neighbors=n_neighbors)
    neighbors.fit(data, labels)
    neighbor_indices = neighbors.kneighbors()[1]
    neighbor_labels = labels[neighbor_indices]
    
    neighborhood_hit_scores = []
    for i in range(0,len(labels)):
        nh_score_i = (neighbor_labels[i] == labels[i]).mean()
        neighborhood_hit_scores.append(nh_score_i)
    
    return np.array(neighborhood_hit_scores).mean()


# executing the NH function
average_nh = neighborhood_hit(data = data, labels = labels, n_neighbors = 5)


##############################################################################

# trying it out on one example

# neighbors = KNeighborsClassifier(n_neighbors=5)
# neighbors.fit(data, labels)

# neighbors.kneighbors()[1]
# neighbor_indices = neighbors.kneighbors()[1]

# neighbor_indices[0].shape

# # extract labels
# labels[neighbor_indices[0]]

# # how many labels are the same as the original point
# labels[0]

# (np.array([1,1,1,1,5]) == 1).mean()

##############################################################################

#                          Thesis Experiments

##############################################################################

data = np.load('./Desktop/Master Thesis/Python experiments/THESIS FINAL RESULTS/Subsets/Sem Seg/Conv8/Baseline/semseg_conv8_UMAP_embedding_hp_0.5_300_rs_1.npy')
labels = np.load('./Desktop/Master Thesis/Python experiments/THESIS FINAL RESULTS/Subsets/Sem Seg/Conv8/Baseline/semseg_predictions.npy')
average_nh = neighborhood_hit(data = data, labels = labels, n_neighbors = 435*5)



##############################################################################


# getting the average number of points in each part for partseg

# run conv9 chamfer script and then this


part_count = []
for i in conv8_hidden_output_baseline_subset:
    inner_list = []
    for j in i:
        inner_list.append(j.shape[0])
    part_count.append(inner_list)


part_count_flat = [item for sublist in part_count for item in sublist]

np.mean(part_count_flat)

# ignore 0s in the mean
part_count_flat = np.array(part_count_flat, dtype = 'float32')
part_count_flat[part_count_flat == 0] = np.nan
np.nanmean(part_count_flat)

# getting the average number of parts per example for partseg (via part labels)


# graph
data = np.load('./Desktop/Master Thesis/Python experiments/THESIS FINAL RESULTS/Subsets/Part Seg/Conv9/Graph/conv9_graph_embedding_labels_2048_1.npy')
labels = np.load('./Desktop/Master Thesis/Python experiments/THESIS FINAL RESULTS/Subsets/Part Seg/Conv9/Graph/graph_labels_parts.npy')
average_nh = neighborhood_hit(data = data, labels = labels, n_neighbors = 3)


