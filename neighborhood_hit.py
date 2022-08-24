# Example on the semantic scene segmentation baseline

# package imports
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.neighbors import KNeighborsClassifier

# setting seed
np.random.seed(1)

# loading the required data
embedding = np.load('semseg_conv8_baseline_embedding_2048_100.npy')
predictions = th.load('semseg_test_predictions_2048_100.pt')
predictions = predictions.permute(0, 2, 1).contiguous()
predictions = predictions.max(dim=2)[1]
predictions = predictions.detach().cpu().numpy()
predictions = np.resize(predictions, (204800,1)).flatten()

# defining a function for calculating the Neighborhood Hit (NH)
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

# calculated average NH (%)
average_nh = neighborhood_hit(data = embedding, labels = predictions, n_neighbors = 435*5)
