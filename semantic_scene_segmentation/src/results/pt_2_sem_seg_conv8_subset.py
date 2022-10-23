# library imports
import torch as th
import numpy as np
import umap
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from scipy.sparse import csr_matrix, csgraph
from umap.umap_ import compute_membership_strengths, smooth_knn_dist, make_epochs_per_sample, simplicial_set_embedding, find_ab_params
import scipy
#%matplotlib inline

# importing utils functions
module_path = os.path.abspath(os.path.join('.\GitHub\master-thesis\semantic_scene_segmentation'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.utils_sem_seg import data_loader_sem_seg, class_labeler, projection, visualize_projection, colormap

# function to extract the UMAP graph representing the high dimensional data
def graph_extraction_umap(hd_data, mindist = 0.5, neighbors = 300, rs = 1):
    # defining the UMAP reducer
    reducer = umap.UMAP(min_dist = mindist, n_neighbors = neighbors, random_state = rs) # this is the original baseline
    # applying UMAP to reduce the dimensionality to 2
    embedding = reducer.fit_transform(hd_data)
    # extracting the graph representing the high dimensional data
    graph_hd_umap = reducer.graph_
    
    return graph_hd_umap

# function to create an aggregated graph from the original high dimensional UMAP graph
def graph_aggregation(class_vector, hd_graph, cloud_size = 2048):

    # creating an index list denoting the first point of each respective object
    examples = []
    for i in range(len(class_vector)):
        if i % cloud_size == 0:
            examples.append(i)
    # denoting the start indices from the second example
    examples_next = examples[1:]
    
    # using the indices to create point range sets per example 
    example_ranges = []
    for j,k in zip(examples, examples_next):
        example_ranges.append(range(j,k))
    # appending the last example range to the list
    example_ranges.append(range(len(class_vector)-cloud_size, len(class_vector)))

    # obtaining the indices per example object for aggregating graph nodes
    object_indices_outer_list = []
    object_outer_list = []
    # iterating over all example ranges
    for ran in example_ranges:
        object_indices_list = []
        object_list = []
        # iterating over all the objects in the range of an example and getting 
        # the indices of the rows containing the object (via predictions/labels)
        for object in np.unique(class_vector[ran[0]:ran[-1]+1]):
            object_indices = np.array(class_vector == object)
            object_indices[0:ran[0]] = False
            object_indices[ran[-1]+1:len(class_vector)] = False
            object_indices_list.append(object_indices)
            object_list.append(object)
        object_indices_outer_list.append(object_indices_list)
        object_outer_list.append(object_list)
    # flattening out the  tensors
    object_indices_outer_list_flattened = [object for sublist in object_indices_outer_list for object in sublist]
    object_outer_list_flattened = [object for sublist in object_outer_list for object in sublist]
    
    # aggregating the graph nodes
    mean_weights_list = []
    for i in range(len(object_outer_list_flattened)):
        inner_list = []
        # obtaining the nodes for one object
        nodes_object = hd_graph[object_indices_outer_list_flattened[i].nonzero()[0],:]
        for j in range(len(object_outer_list_flattened)):
            if i != j:
                # obtaining the connected nodes with another object
                nodes_connected = nodes_object.transpose()[object_indices_outer_list_flattened[j].nonzero()[0],:]
                # summing up the weights between the nodes of two objects
                sum_weights = nodes_connected.sum()
                # averaging the weights
                mean_weights = sum_weights/nodes_connected.count_nonzero()
                inner_list.append(mean_weights) 
            else:
                # intersection with identical object excluded
                inner_list.append(0) # to append 0 at those positions
        mean_weights_list.append(inner_list)
    # converting to a numpy array
    mean_weights_list_numpy = np.array(mean_weights_list)
    # removing NaN values
    mean_weights_list_numpy = np.nan_to_num(mean_weights_list_numpy)
    # converting to a scipy sparse matrix
    graph_aggregated = csr_matrix(mean_weights_list_numpy)
    
    return graph_aggregated, object_outer_list_flattened

# function to carry out the UMAP opimization in order to obtain the embedding (projection)
# source: https://umap-learn.readthedocs.io/en/latest/api.html
def umap_embedding_pt_2(graph, metric="euclidean", n_components = 2, n_epochs = 500, 
                        spread=1.0, min_dist = 1, initial_alpha=1.0, 
                        negative_sample_rate=5, repulsion_strength=1.0):
    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]
    if n_epochs <= 0:
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200
    # optimize embedding graph     
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

# execution
if __name__ == '__main__':
    conv8_hidden_output_subset, preds_subset, labels_subset = data_loader_sem_seg()
    # extracting the UMAP graph representing the high dimensional data
    graph_sparse = graph_extraction_umap(conv8_hidden_output_subset)
    # compute the aggregated graph representing the high dimensional data
    graph_agg, graph_object_reference = graph_aggregation(preds_subset, graph_sparse)
    # use the aggregated graph to compute the PT-2 UMAP embedding
    embedding_pt_2 = umap_embedding_pt_2(graph_agg)
    # preprocessing prediction array for visualization
    preds_subset_strings = class_labeler(graph_object_reference)
    # visualizing the PT-2 2D embedding (colored by predictions)
    visualize_projection(embedding_pt_2, preds_subset_strings, colormap,
                         title = 'UMAP PT-2 projection (colored via predictions) \n on a test subset of 100 samples (cloud size = 2048)')