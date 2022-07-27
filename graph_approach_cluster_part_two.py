# library imports
import torch as th
import numpy as np
import umap
#import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd

from scipy.sparse import csr_matrix, csgraph
from umap.umap_ import compute_membership_strengths, smooth_knn_dist, make_epochs_per_sample, simplicial_set_embedding, find_ab_params
import scipy
#from sklearn.preprocessing import StandardScaler
#from sklearn.neighbors import NearestNeighbors

# loading the 2D embedding and UMAP graph
graph_new = scipy.sparse.load_npz('./Results/partseg_full_data_conv9/conv9_protocol_two_graph_300_0.5_rs_1.npz')
#embedding = np.load('./Data/partseg_full_data_conv9/conv9_UMAP_embedding_hp_0.5_300_rs_1.npy')

def transform(graph, metric="euclidean", n_components = 2, n_epochs = 500, 
              spread=1.0, min_dist = 1, initial_alpha=1.0, 
              negative_sample_rate=5, repulsion_strength=1.0):
    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]

    if n_epochs <= 0:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200

    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()

    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

    head = graph.row
    tail = graph.col
    weight = graph.data

    a, b = find_ab_params(spread, min_dist)
    
    emebedding = simplicial_set_embedding(None, graph, n_components=2, 
                                          initial_alpha=1.0, a=a, b=b, 
                                          gamma=repulsion_strength, 
                                          negative_sample_rate=negative_sample_rate, 
                                          random_state=np.random.RandomState(seed=1),
                                          metric=metric, 
                                          metric_kwds=None, verbose=False, 
                                          parallel=False, n_epochs=n_epochs, 
                                          init="spectral", densmap = False,
                                          output_dens = False,
                                          densmap_kwds = {"mnr": 'abc'})
    
    return emebedding


final_output_protocol_two = transform(graph_new)
final_output_protocol_two = final_output_protocol_two[0]

np.save('./Results/conv9_protocol_two_embedding_full_data_hp_0.5_300_rs_1.npy', final_output_protocol_two)