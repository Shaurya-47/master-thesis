# library imports
import torch as th
import numpy as np
import umap
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import csr_matrix, csgraph
from umap.umap_ import compute_membership_strengths, smooth_knn_dist, make_epochs_per_sample, simplicial_set_embedding, find_ab_params
import scipy

# importing data
airplane_test_subset_conv9_hidden_output = th.load('airplane_test_subset_conv9_hidden_output.pt')
bag_test_subset_conv9_hidden_output = th.load('bag_test_subset_conv9_hidden_output.pt')
cap_test_subset_conv9_hidden_output = th.load('cap_test_subset_conv9_hidden_output.pt')
car_test_subset_conv9_hidden_output = th.load('car_test_subset_conv9_hidden_output.pt')
chair_test_subset_conv9_hidden_output = th.load('chair_test_subset_conv9_hidden_output.pt')
earphone_test_subset_conv9_hidden_output = th.load('earphone_test_subset_conv9_hidden_output.pt')
guitar_test_subset_conv9_hidden_output = th.load('guitar_test_subset_conv9_hidden_output.pt')
knife_test_subset_conv9_hidden_output = th.load('knife_test_subset_conv9_hidden_output.pt')
lamp_test_subset_conv9_hidden_output = th.load('lamp_test_subset_conv9_hidden_output.pt')
laptop_test_subset_conv9_hidden_output = th.load('laptop_test_subset_conv9_hidden_output.pt')
motorbike_test_subset_conv9_hidden_output = th.load('motorbike_test_subset_conv9_hidden_output.pt')
mug_test_subset_conv9_hidden_output = th.load('mug_test_subset_conv9_hidden_output.pt')
pistol_test_subset_conv9_hidden_output = th.load('pistol_test_subset_conv9_hidden_output.pt')
rocket_test_subset_conv9_hidden_output = th.load('rocket_test_subset_conv9_hidden_output.pt')
skateboard_test_subset_conv9_hidden_output = th.load('skateboard_test_subset_conv9_hidden_output.pt')
table_test_subset_conv9_hidden_output = th.load('table_test_subset_conv9_hidden_output.pt')

airplane_test_subset_part_labels = th.load('airplane_test_subset_part_labels.pt')
bag_test_subset_part_labels = th.load('bag_test_subset_part_labels.pt')
cap_test_subset_part_labels = th.load('cap_test_subset_part_labels.pt')
car_test_subset_part_labels = th.load('car_test_subset_part_labels.pt')
chair_test_subset_part_labels = th.load('chair_test_subset_part_labels.pt')
earphone_test_subset_part_labels = th.load('earphone_test_subset_part_labels.pt')
guitar_test_subset_part_labels = th.load('guitar_test_subset_part_labels.pt')
knife_test_subset_part_labels = th.load('knife_test_subset_part_labels.pt')
lamp_test_subset_part_labels = th.load('lamp_test_subset_part_labels.pt')
laptop_test_subset_part_labels = th.load('laptop_test_subset_part_labels.pt')
motorbike_test_subset_part_labels = th.load('motorbike_test_subset_part_labels.pt')
mug_test_subset_part_labels = th.load('mug_test_subset_part_labels.pt')
pistol_test_subset_part_labels = th.load('pistol_test_subset_part_labels.pt')
rocket_test_subset_part_labels = th.load('rocket_test_subset_part_labels.pt')
skateboard_test_subset_part_labels = th.load('skateboard_test_subset_part_labels.pt')
table_test_subset_part_labels = th.load('table_test_subset_part_labels.pt')

airplane_test_subset_predictions = th.load('airplane_test_subset_predictions.pt')
bag_test_subset_predictions = th.load('bag_test_subset_predictions.pt')
cap_test_subset_predictions = th.load('cap_test_subset_predictions.pt')
car_test_subset_predictions = th.load('car_test_subset_predictions.pt')
chair_test_subset_predictions = th.load('chair_test_subset_predictions.pt')
earphone_test_subset_predictions = th.load('earphone_test_subset_predictions.pt')
guitar_test_subset_predictions = th.load('guitar_test_subset_predictions.pt')
knife_test_subset_predictions = th.load('knife_test_subset_predictions.pt')
lamp_test_subset_predictions = th.load('lamp_test_subset_predictions.pt')
laptop_test_subset_predictions = th.load('laptop_test_subset_predictions.pt')
motorbike_test_subset_predictions = th.load('motorbike_test_subset_predictions.pt')
mug_test_subset_predictions = th.load('mug_test_subset_predictions.pt')
pistol_test_subset_predictions = th.load('pistol_test_subset_predictions.pt')
rocket_test_subset_predictions = th.load('rocket_test_subset_predictions.pt')
skateboard_test_subset_predictions = th.load('skateboard_test_subset_predictions.pt')
table_test_subset_predictions = th.load('table_test_subset_predictions.pt')

# dropping the batch size dimension of the tensors
airplane_conv9_hidden_output_flipped = np.moveaxis(airplane_test_subset_conv9_hidden_output, 2, 1)
airplane_conv9_hidden_output_resized = np.resize(airplane_conv9_hidden_output_flipped, (100*2048,256))
bag_conv9_hidden_output_flipped = np.moveaxis(bag_test_subset_conv9_hidden_output, 2, 1)
bag_conv9_hidden_output_resized = np.resize(bag_conv9_hidden_output_flipped, (14*2048,256))
cap_conv9_hidden_output_flipped = np.moveaxis(cap_test_subset_conv9_hidden_output, 2, 1)
cap_conv9_hidden_output_resized = np.resize(cap_conv9_hidden_output_flipped, (11*2048,256))
car_conv9_hidden_output_flipped = np.moveaxis(car_test_subset_conv9_hidden_output, 2, 1)
car_conv9_hidden_output_resized = np.resize(car_conv9_hidden_output_flipped, (100*2048,256))
chair_conv9_hidden_output_flipped = np.moveaxis(chair_test_subset_conv9_hidden_output, 2, 1)
chair_conv9_hidden_output_resized = np.resize(chair_conv9_hidden_output_flipped, (100*2048,256))
earphone_conv9_hidden_output_flipped = np.moveaxis(earphone_test_subset_conv9_hidden_output, 2, 1)
earphone_conv9_hidden_output_resized = np.resize(earphone_conv9_hidden_output_flipped, (14*2048,256))
guitar_conv9_hidden_output_flipped = np.moveaxis(guitar_test_subset_conv9_hidden_output, 2, 1)
guitar_conv9_hidden_output_resized = np.resize(guitar_conv9_hidden_output_flipped, (100*2048,256))
knife_conv9_hidden_output_flipped = np.moveaxis(knife_test_subset_conv9_hidden_output, 2, 1)
knife_conv9_hidden_output_resized = np.resize(knife_conv9_hidden_output_flipped, (80*2048,256))
lamp_conv9_hidden_output_flipped = np.moveaxis(lamp_test_subset_conv9_hidden_output, 2, 1)
lamp_conv9_hidden_output_resized = np.resize(lamp_conv9_hidden_output_flipped, (100*2048,256))
laptop_conv9_hidden_output_flipped = np.moveaxis(laptop_test_subset_conv9_hidden_output, 2, 1)
laptop_conv9_hidden_output_resized = np.resize(laptop_conv9_hidden_output_flipped, (83*2048,256))
motorbike_conv9_hidden_output_flipped = np.moveaxis(motorbike_test_subset_conv9_hidden_output, 2, 1)
motorbike_conv9_hidden_output_resized = np.resize(motorbike_conv9_hidden_output_flipped, (51*2048,256))
mug_conv9_hidden_output_flipped = np.moveaxis(mug_test_subset_conv9_hidden_output, 2, 1)
mug_conv9_hidden_output_resized = np.resize(mug_conv9_hidden_output_flipped, (38*2048,256))
pistol_conv9_hidden_output_flipped = np.moveaxis(pistol_test_subset_conv9_hidden_output, 2, 1)
pistol_conv9_hidden_output_resized = np.resize(pistol_conv9_hidden_output_flipped, (44*2048,256))
rocket_conv9_hidden_output_flipped = np.moveaxis(rocket_test_subset_conv9_hidden_output, 2, 1)
rocket_conv9_hidden_output_resized = np.resize(rocket_conv9_hidden_output_flipped, (12*2048,256))
skateboard_conv9_hidden_output_flipped = np.moveaxis(skateboard_test_subset_conv9_hidden_output, 2, 1)
skateboard_conv9_hidden_output_resized = np.resize(skateboard_conv9_hidden_output_flipped, (31*2048,256))
table_conv9_hidden_output_flipped = np.moveaxis(table_test_subset_conv9_hidden_output, 2, 1)
table_conv9_hidden_output_resized = np.resize(table_conv9_hidden_output_flipped, (100*2048,256))

# combining the subsets
conv9_hidden_output_subset = np.vstack((airplane_conv9_hidden_output_resized, 
                                        bag_conv9_hidden_output_resized,
                                        cap_conv9_hidden_output_resized,
                                        car_conv9_hidden_output_resized,
                                        chair_conv9_hidden_output_resized,
                                        earphone_conv9_hidden_output_resized,
                                        guitar_conv9_hidden_output_resized,
                                        knife_conv9_hidden_output_resized,
                                        lamp_conv9_hidden_output_resized,
                                        laptop_conv9_hidden_output_resized,
                                        motorbike_conv9_hidden_output_resized,
                                        mug_conv9_hidden_output_resized,
                                        pistol_conv9_hidden_output_resized,
                                        rocket_conv9_hidden_output_resized,
                                        skateboard_conv9_hidden_output_resized,
                                        table_conv9_hidden_output_resized
                                        ))
print(conv9_hidden_output_subset.shape)

# defining the UMAP reducer
reducer = umap.UMAP(min_dist = 0.5, n_neighbors = 300, random_state = 1) # this is the original baseline
# applying UMAP to reduce the dimensionality to 2
embedding = reducer.fit_transform(conv9_hidden_output_subset)
#print(embedding.shape)

# saving the UMAP graph from the embedding
np.save('./Results/conv9_full_data_baseline_UMAP_embedding_hp_0.5_300_rs_1.npy', embedding)
scipy.sparse.save_npz('conv8_graph_hp_0.5_300_rs_1.npz', reducer.graph_)
graph_sparse = reducer.graph_

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
predictions_subset = np.resize(predictions_subset, (163840,1)).flatten()
np.unique(predictions_subset)
part_labels_subset = np.resize(part_labels_subset, (163840,1)).flatten()
np.unique(part_labels_subset)

# creating an index list for the start of each example
examples = []
for i in range(len(predictions_subset)):
    if i % 1024 == 0:
        examples.append(i)
examples_forward = examples[1:]

# creating the range sets per example 
example_ranges = []
for j,k in zip(examples, examples_forward):
    example_ranges.append(range(j,k))

# appending the last index to the list
example_ranges.append(range(len(predictions_subset)-1024, len(predictions_subset)))

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

np.save('./Results/conv9_protocol_two_embedding_hp_0.5_300_rs_1.npy', final_output_protocol_two)

# plot pre-requisites 
part_outer_list_partmap = part_outer_list_flattened.copy()

# pandas mapping approach
part_outer_list_partmap = ["airplane_0" if x==0 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["airplane_1" if x==1 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["airplane_2" if x==2 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["airplane_3" if x==3 else x for x in part_outer_list_partmap]

part_outer_list_partmap = ["bag_0" if x==4 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["bag_1" if x==5 else x for x in part_outer_list_partmap]

part_outer_list_partmap = ["cap_0" if x==6 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["cap_1" if x==7 else x for x in part_outer_list_partmap]

part_outer_list_partmap = ["car_0" if x==8 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["car_1" if x==9 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["car_2" if x==10 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["car_3" if x==11 else x for x in part_outer_list_partmap]

part_outer_list_partmap = ["chair_0" if x==12 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["chair_1" if x==13 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["chair_2" if x==14 else x for x in part_outer_list_partmap]
                         # part skip
part_outer_list_partmap = ["earphone_0" if x==16 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["earphone_1" if x==17 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["earphone_2" if x==18 else x for x in part_outer_list_partmap]

part_outer_list_partmap = ["guitar_0" if x==19 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["guitar_1" if x==20 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["guitar_2" if x==21 else x for x in part_outer_list_partmap]

part_outer_list_partmap = ["knife_0" if x==22 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["knife_1" if x==23 else x for x in part_outer_list_partmap]

part_outer_list_partmap = ["lamp_0" if x==24 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["lamp_1" if x==25 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["lamp_2" if x==26 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["lamp_3" if x==27 else x for x in part_outer_list_partmap]

part_outer_list_partmap = ["laptop_0" if x==28 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["laptop_1" if x==29 else x for x in part_outer_list_partmap]
                         
part_outer_list_partmap = ["motor_0" if x==30 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["motor_1" if x==31 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["motor_2" if x==32 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["motor_3" if x==33 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["motor_4" if x==34 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["motor_5" if x==35 else x for x in part_outer_list_partmap]

part_outer_list_partmap = ["mug_0" if x==36 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["mug_1" if x==37 else x for x in part_outer_list_partmap]

part_outer_list_partmap = ["pistol_0" if x==38 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["pistol_1" if x==39 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["pistol_2" if x==40 else x for x in part_outer_list_partmap]

part_outer_list_partmap = ["rocket_0" if x==41 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["rocket_1" if x==42 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["rocket_2" if x==43 else x for x in part_outer_list_partmap]

part_outer_list_partmap = ["skateboard_0" if x==44 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["skateboard_1" if x==45 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["skateboard_2" if x==46 else x for x in part_outer_list_partmap]

part_outer_list_partmap = ["table_0" if x==47 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["table_1" if x==48 else x for x in part_outer_list_partmap]
part_outer_list_partmap = ["table_2" if x==49 else x for x in part_outer_list_partmap]

part_outer_list_partmap = np.array(part_outer_list_partmap).flatten()
part_outer_list_partmap = pd.Series(part_outer_list_partmap)
part_outer_list_partmap.shape
#print(part_outer_list_partmap.unique())

colormap_dict = {"airplane_0": "#CFD8DC", "airplane_1": "#90A4AE", "airplane_2": "#607D8B", "airplane_3": "#455A64",
                "bag_0": "#BDBDBD", "bag_1": "#616161",
                "cap_0": "#FF8A65", "cap_1": "#E64A19",
                "car_0": "#BCAAA4", "car_1": "#8D6E63", "car_2": "#6D4C41", "car_3": "#4E342E",
                "chair_0": "#FFE0B2", "chair_1": "#FFB74D", "chair_2": "#FB8C00", #, "olivedrab", # chair part
                "earphone_0": "#FFF9C4", "earphone_1": "#FFF176", "earphone_2": "#FDD835",
                "guitar_0": "#DCE775", "guitar_1": "#C0CA33", "guitar_2": "#9E9D24",
                "knife_0": "#4CAF50", "knife_1": "#1B5E20",
                "lamp_0": "#80CBC4", "lamp_1": "#26A69A", "lamp_2": "#00897B", "lamp_3": "#004D40",
                "laptop_0": "#80DEEA", "laptop_1": "#00BCD4",
                "motor_0": "#BBDEFB", "motor_1": "#90CAF9", "motor_2": "#64B5F6", "motor_3": "#2196F3", "motor_4": "#1976D2", "motor_5": "#0D47A1",#  "mediumpurple" # secondlast color (motorbike part)
                "mug_0": "#3F51B5", "mug_1": "#1A237E",
                "pistol_0": "#B39DDB", "pistol_1": "#7E57C2", "pistol_2": "#512DA8",
                "rocket_0": "#EF9A9A", "rocket_1": "#EF5350", "rocket_2": "#C62828",
                "skateboard_0": "#F8BBD0", "skateboard_1": "#F06292", "skateboard_2": "#E91E63",
                "table_0": "#E1BEE7", "table_1": "#BA68C8", "table_2": "#9C27B0"
                }
    
colormap = part_outer_list_partmap.map(colormap_dict)

# plotting
plt.figure(figsize=(8,6), dpi=800)
plt.scatter(final_output_protocol_two[:,0], final_output_protocol_two[:,1], s=17, marker = 'v', c = colormap)
plt.title('UMAP PT-2 projection (using predictions) \n on 10 samples (size 1024) from all 16 categories', fontsize=24)
