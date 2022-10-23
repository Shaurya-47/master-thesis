# importing libraries
import numpy as np
import pandas as pd
import torch as th
import umap
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from pytorch3d.loss import chamfer_distance as cd

# function to load in the S3DIS subset data for Conv layer 8 (100 examples of size 2048 each)
def data_loader_sem_seg():
    conv8_hidden_output_subset = th.load('./semantic_scene_segmentation/data/conv8_hidden_output_subset.pt')
    preds_subset = th.load('./semantic_scene_segmentation/data/preds_subset.pt')
    labels_subset = th.load('./semantic_scene_segmentation/data/labels_subset.pt')
    
    return conv8_hidden_output_subset, preds_subset, labels_subset

# function to obtain a UMAP 2D embedding from high dimensional data
def projection(hd_data, mindist = 0.5, neighbors = 300, rs = 1):
    # defining the UMAP reducer
    reducer = umap.UMAP(min_dist = mindist, n_neighbors = neighbors, random_state = rs) # this is the original baseline
    # applying UMAP to reduce the dimensionality to 2
    embedding = reducer.fit_transform(hd_data)
    
    return embedding

# function to visualize the UMAP embedding
def visualize_projection(embedding_input, string_labels, color_map, title):
    plt.figure(figsize=(8,6), dpi=1000)
    plt.scatter(embedding_input[:,0], embedding_input[:,1], s=0.05,
                c=string_labels.map(color_map))
    plt.title(title, fontsize=24)
    plt.show()
    
# function to visualize the UMAP embedding with points colored manually
def visualize_projection_manual_color(embedding_input, color_list, title):
    plt.figure(figsize=(8,6), dpi=1000)
    plt.scatter(embedding_input[:,0], embedding_input[:,1], s=0.05,
                c=color_list)
    plt.title(title, fontsize=24)
    plt.show()

# function to calculate the average neighborhood hit (NH) score across a set of points
def neighborhood_hit(data, labels, n_neighbors):
    
    # getting the KNN indices for each point in the dataset
    neighbors = KNeighborsClassifier(n_neighbors=n_neighbors)
    neighbors.fit(data, labels)
    neighbor_indices = neighbors.kneighbors()[1]
    neighbor_labels = labels[neighbor_indices]
    
    # computing the scores per point using prediction or label information
    neighborhood_hit_scores = []
    for i in range(0,len(labels)):
        nh_score_i = (neighbor_labels[i] == labels[i]).mean()
        neighborhood_hit_scores.append(nh_score_i)
    
    # returning the average NH score across all points
    return np.array(neighborhood_hit_scores).mean()
    
# function to constrct a CD matrix from a collection of input sets (GPU version)
def construct_cd_matrix(cd_input, layer_dimension = 256):
    # computing CD
    cd_matrix = np.asarray([[list(cd(th.from_numpy(np.resize(p1, (1,p1.shape[0],layer_dimension))).cuda(), th.from_numpy(np.resize(p2, (1,p2.shape[0],layer_dimension))).cuda()))[0].cpu() for p2 in cd_input] for p1 in cd_input])
    # post-processing
    cd_matrix = np.vectorize(lambda x: x.item())(cd_matrix)
    # drop NaN rows
    cd_matrix = cd_matrix.flatten()
    cd_matrix = cd_matrix[~np.isnan(cd_matrix)]
    # shifting back to matrix form
    cd_matrix = np.reshape(cd_matrix, (np.sqrt(cd_matrix), np.sqrt(cd_matrix)))
    
    return cd_matrix

# defined color map for semantic segmentation
colormap = {"ceiling": "#9E9E9E", # grey
            "floor": "#795548", # brown
            "wall": "#FF5722", # dark orange
            "beam": "#FFC107",  # amber(dark yellow)
            "column": "#FFEE58", # light yellow
            "window": "#CDDC39", # lime
            "door": "#4CAF50", # green
            "table": "#009688", # table
            "chair": "#00BCD4" ,# cyan-light blue
            "sofa": "#2196F3", # blue
            "bookcase": "#3F51B5", # indigo(dark blue)
            "board": "#9C27B0", # purple
            "clutter": "#E91E63", # pink
            }

# function to arrign string labels to model predictions/dataset labels
def class_labeler(input_array):
    
    # converting to list
    input_array = input_array.tolist()
    
    # adding string class labels 
    input_array = input_array.tolist()
    input_array = ["ceiling" if x==0 else x for x in input_array]
    input_array = ["floor" if x==1 else x for x in input_array]
    input_array = ["wall" if x==2 else x for x in input_array]
    input_array = ["beam" if x==3 else x for x in input_array]
    input_array = ["column" if x==4 else x for x in input_array]
    input_array = ["window" if x==5 else x for x in input_array]
    input_array = ["door" if x==6 else x for x in input_array]
    input_array = ["table" if x==7 else x for x in input_array]
    input_array = ["chair" if x==8 else x for x in input_array]
    input_array = ["sofa" if x==9 else x for x in input_array]
    input_array = ["bookcase" if x==10 else x for x in input_array]
    input_array = ["board" if x==11 else x for x in input_array]
    input_array = ["clutter" if x==12 else x for x in input_array]
    
    # converting to a pandas series
    input_array = np.array(input_array).flatten()
    input_array = pd.Series(input_array)
    
    return input_array

# function to prepare chamfer distance (CD) input for the S3DIS dataset
def prepare_cd_input_s3dis(hidden_layer_output, predictions):
    # creating boolean masks per object and per example
    indices_ceiling = np.array(predictions == 0)
    indices_floor = np.array(predictions == 1)
    indices_wall = np.array(predictions == 2)
    indices_beam = np.array(predictions == 3)
    indices_column = np.array(predictions == 4)
    indices_window = np.array(predictions == 5)
    indices_door = np.array(predictions == 6)
    indices_table = np.array(predictions == 7)
    indices_chair = np.array(predictions == 8)
    indices_sofa = np.array(predictions == 9)
    indices_bookcase = np.array(predictions == 10)
    indices_board = np.array(predictions == 11)
    indices_clutter = np.array(predictions == 12)
    
    # applying object masks to get array subsets per example object
    # creating placeholders
    hidden_layer_output_ceiling = []
    hidden_layer_output_floor = []
    hidden_layer_output_wall = []
    hidden_layer_output_beam = []
    hidden_layer_output_column = []
    hidden_layer_output_window = []
    hidden_layer_output_door = []
    hidden_layer_output_table = []
    hidden_layer_output_chair = []
    hidden_layer_output_sofa = []
    hidden_layer_output_bookcase = []
    hidden_layer_output_board = []
    hidden_layer_output_clutter = []
    
    # main loop for object subsetting per example
    for i in range(predictions.shape[0]):
        # subsetting by example objects
        inner_result_ceiling = hidden_layer_output[i][indices_ceiling[i]]
        inner_result_floor = hidden_layer_output[i][indices_floor[i]]
        inner_result_wall = hidden_layer_output[i][indices_wall[i]]
        inner_result_beam = hidden_layer_output[i][indices_beam[i]]
        inner_result_column = hidden_layer_output[i][indices_column[i]]
        inner_result_window = hidden_layer_output[i][indices_window[i]]
        inner_result_door = hidden_layer_output[i][indices_door[i]]
        inner_result_table = hidden_layer_output[i][indices_table[i]]
        inner_result_chair = hidden_layer_output[i][indices_chair[i]]
        inner_result_sofa = hidden_layer_output[i][indices_sofa[i]]
        inner_result_bookcase = hidden_layer_output[i][indices_bookcase[i]]
        inner_result_board = hidden_layer_output[i][indices_board[i]]
        inner_result_clutter = hidden_layer_output[i][indices_clutter[i]]
            
        # appending to lists
        hidden_layer_output_ceiling.append(inner_result_ceiling)
        hidden_layer_output_floor.append(inner_result_floor)
        hidden_layer_output_wall.append(inner_result_wall)
        hidden_layer_output_beam.append(inner_result_beam)
        hidden_layer_output_column.append(inner_result_column
        hidden_layer_output_window.append(inner_result_window)
        hidden_layer_output_door.append(inner_result_door)
        hidden_layer_output_table.append(inner_result_table)
        hidden_layer_output_chair.append(inner_result_chair)
        hidden_layer_output_sofa.append(inner_result_sofa)
        hidden_layer_output_bookcase.append(inner_result_bookcase)
        hidden_layer_output_board.append(inner_result_board)
        hidden_layer_output_clutter.append(inner_result_clutter)
        
    # stacking all example object sets together - input for Chamfer Distance
    # (window and sofa objects do not exist in the data; omitting)
    chamfer_distance_input = np.vstack((hidden_layer_output_ceiling,
                                        hidden_layer_output_floor,
                                        hidden_layer_output_wall,
                                        hidden_layer_output_beam,
                                        hidden_layer_output_column,
                                        hidden_layer_output_door,
                                        hidden_layer_output_table,
                                        hidden_layer_output_chair,
                                        hidden_layer_output_bookcase,
                                        hidden_layer_output_board,
                                        hidden_layer_output_clutter))
    
    # flattening the tensor
    cd_input = np.resize(chamfer_distance_input,
                         (chamfer_distance_input.shape[0]*chamfer_distance_input.shape[1],1)).flatten()
    
    return cd_input

# defining the color list for the PT-1 output on the S3DIS subset
color_list_pt_1_conv8_subset_s3dis = []
# ceiling
color_list_pt_1_conv8_subset_s3dis.append(['#9E9E9E'] * 92)
# floor
color_list_pt_1_conv8_subset_s3dis.append(['#795548'] * 97)
# wall
color_list_pt_1_conv8_subset_s3dis.append(['#FF5722'] * 69)
# beam
color_list_pt_1_conv8_subset_s3dis.append(['#FFC107'] * 35)
# column
color_list_pt_1_conv8_subset_s3dis.append(['#FFEE58'] * 14)
# door
color_list_pt_1_conv8_subset_s3dis.append(['#4CAF50'] * 42)
# table
color_list_pt_1_conv8_subset_s3dis.append(['#009688'] * 14)
# chair
color_list_pt_1_conv8_subset_s3dis.append(['#00BCD4'] * 14)
# bookcase
color_list_pt_1_conv8_subset_s3dis.append(['#3F51B5'] * 11)
# board
color_list_pt_1_conv8_subset_s3dis.append(['#9C27B0'] * 6)
# clutter
color_list_pt_1_conv8_subset_s3dis.append(['#E91E63'] * 77)
# post-processing
color_list_pt_1_conv8_subset_s3dis = [item for sublist in color_list_pt_1_conv8_subset_s3dis for item in sublist]
color_list_pt_1_conv8_subset_s3dis = pd.Series(color_list_pt_1_conv8_subset_s3dis)

