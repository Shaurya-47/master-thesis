# library imports
import torch as th
import numpy as np
import umap
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import csr_matrix, csgraph
from umap.umap_ import compute_membership_strengths, smooth_knn_dist, make_epochs_per_sample, simplicial_set_embedding, find_ab_params
import scipy

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

#running UMAP on the embedding
reducer = umap.UMAP(min_dist = 0.5, n_neighbors = 300, random_state = 1)
embedding = reducer.fit_transform(conv8_output)
np.save('conv8_UMAP_embedding_hp_0.5_300_rs_1.npy', embedding)
scipy.sparse.save_npz('conv8_graph_hp_0.5_300_rs_1.npz', reducer.graph_)
graph_sparse = reducer.graph_

# PT-2 process

# creating an index list for the start of each example
examples = []
for i in range(len(predictions)):
    if i % 2048 == 0:
        examples.append(i)
examples_forward = examples[1:]

# creating the range sets per example 
example_ranges = []
for j,k in zip(examples, examples_forward):
    example_ranges.append(range(j,k))

# appending the last index to the list
example_ranges.append(range(len(predictions)-2048, len(predictions)))

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
    for part in np.unique(predictions[ran[0]:ran[-1]+1]):

        part_indices = np.array(predictions == part)
        part_indices[0:ran[0]] = False
        part_indices[ran[-1]+1:len(predictions)] = False
        part_indices_list.append(part_indices)
  
        part_list.append(part)
        
    part_indices_outer_list.append(part_indices_list)
    part_outer_list.append(part_list)
    
part_indices_outer_list_flattened = [part for sublist in part_indices_outer_list for part in sublist]
part_outer_list_flattened = [part for sublist in part_outer_list for part in sublist]

# FINAL NESTED LOOP
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

def transform(graph, metric="euclidean", n_components = 2, n_epochs = 500, 
              spread=1.0, min_dist = 0.2, initial_alpha=1.0, 
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

# plotting pre-requisites
part_outer_list_partmap = part_outer_list_flattened.copy()
part_outer_list_partmap = ["ceiling" if x==0 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["floor" if x==1 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["wall" if x==2 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["beam" if x==3 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["column" if x==4 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["window" if x==5 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["door" if x==6 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["table" if x==7 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["chair" if x==8 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["sofa" if x==9 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["bookcase" if x==10 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["board" if x==11 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["clutter" if x==12 else x for x in part_outer_list_partmap]
part_outer_list_partmap = np.array(part_outer_list_partmap).flatten()
part_outer_list_partmap = pd.Series(part_outer_list_partmap)
part_outer_list_partmap.shape
print(part_outer_list_partmap.unique())

# color map 
colormap = {"ceiling": "#9E9E9E",#grey
            "floor": "#795548",#brown
            "wall": "#FF5722",#d orange
            "beam": "#FFC107",#amber(dark yellow)
            "column": "#FFEE58",#light yellow
            "window": "#CDDC39",#lime
            "door": "#4CAF50",#green
            "table": "#009688",#table
            "chair": "#00BCD4",#cyan-light blue
            "sofa": "#2196F3",#blue
            "bookcase": "#3F51B5",#indigo(dark blue)
            "board": "#9C27B0",#purple
            "clutter": "#E91E63",#pink
            }
    
colormap = part_outer_list_partmap.map(colormap)
#print(colormap.unique())
# NA detection
#print(colormap[colormap.isnull().any(0)])

# plotting
plt.figure(figsize=(8,6), dpi=800)
plt.scatter(final_output_protocol_two[:,0], final_output_protocol_two[:,1], s=17, marker = 'v', c = colormap)
plt.title('UMAP PT-2 projection (using predictions) \n on 100 samples (cloud size 1024)', fontsize=24)
