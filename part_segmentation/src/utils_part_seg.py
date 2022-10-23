# importing libraries
import numpy as np
import pandas as pd
import torch as th
import umap
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from pytorch3d.loss import chamfer_distance as cd

# function to load in the ShapeNet Part subset data for Conv layer 9 (160 examples of size 1024 each)
def data_loader_part_seg():
    conv9_hidden_output_subset = th.load('./part_segmentation/data/conv9_hidden_output_subset.pt')
    preds_subset = th.load('./part_segmentation/data/preds_subset.pt')
    part_labels_subset = th.load('./part_segmentation/data/part_labels_subset.pt')
    
    return conv9_hidden_output_subset, preds_subset, part_labels_subset

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

# defined color map for part segmentation
colormap = {"airplane_0": "#CFD8DC", "airplane_1": "#90A4AE", "airplane_2": "#607D8B", "airplane_3": "#455A64",
            "bag_0": "#BDBDBD", "bag_1": "#616161",
            "cap_0": "#FF8A65", "cap_1": "#E64A19",
            "car_0": "#BCAAA4", "car_1": "#8D6E63", "car_2": "#6D4C41", "car_3": "#4E342E",
            "chair_0": "#FFE0B2", "chair_1": "#FFB74D", "chair_2": "#FB8C00",
            "earphone_0": "#FFF9C4", "earphone_1": "#FFF176", "earphone_2": "#FDD835",
            "guitar_0": "#DCE775", "guitar_1": "#C0CA33", "guitar_2": "#9E9D24",
            "knife_0": "#4CAF50", "knife_1": "#1B5E20",
            "lamp_0": "#80CBC4", "lamp_1": "#26A69A", "lamp_2": "#00897B", "lamp_3": "#004D40",
            "laptop_0": "#80DEEA", "laptop_1": "#00BCD4",
            "motor_0": "#BBDEFB", "motor_1": "#90CAF9", "motor_2": "#64B5F6", "motor_3": "#2196F3", "motor_4": "#1976D2", "motor_5": "#0D47A1",
            "mug_0": "#3F51B5", "mug_1": "#1A237E",
            "pistol_0": "#B39DDB", "pistol_1": "#7E57C2", "pistol_2": "#512DA8",
            "rocket_0": "#EF9A9A", "rocket_1": "#EF5350", "rocket_2": "#C62828",
            "skateboard_0": "#F8BBD0", "skateboard_1": "#F06292", "skateboard_2": "#E91E63",
            "table_0": "#E1BEE7", "table_1": "#BA68C8", "table_2": "#9C27B0"
            }

# function to arrign string labels to model predictions/dataset labels
def class_labeler(input_array):
    
    # converting to list
    input_array = input_array.tolist()
    
    # adding string class labels 
    input_array = ["airplane_0" if x==0 else x for x in input_array]
    input_array = ["airplane_1" if x==1 else x for x in input_array]
    input_array = ["airplane_2" if x==2 else x for x in input_array]
    input_array = ["airplane_3" if x==3 else x for x in input_array]
    input_array = ["bag_0" if x==4 else x for x in input_array]
    input_array = ["bag_1" if x==5 else x for x in input_array]
    input_array = ["cap_0" if x==6 else x for x in input_array]
    input_array = ["cap_1" if x==7 else x for x in input_array]
    input_array = ["car_0" if x==8 else x for x in input_array]
    input_array = ["car_1" if x==9 else x for x in input_array]
    input_array = ["car_2" if x==10 else x for x in input_array]
    input_array = ["car_3" if x==11 else x for x in input_array]
    input_array = ["chair_0" if x==12 else x for x in input_array]
    input_array = ["chair_1" if x==13 else x for x in input_array]
    input_array = ["chair_2" if x==14 else x for x in input_array]
    input_array = ["earphone_0" if x==16 else x for x in input_array]
    input_array = ["earphone_1" if x==17 else x for x in input_array]
    input_array = ["earphone_2" if x==18 else x for x in input_array]
    input_array = ["guitar_0" if x==19 else x for x in input_array]
    input_array = ["guitar_1" if x==20 else x for x in input_array]
    input_array = ["guitar_2" if x==21 else x for x in input_array]
    input_array = ["knife_0" if x==22 else x for x in input_array]
    input_array = ["knife_1" if x==23 else x for x in input_array]
    input_array = ["lamp_0" if x==24 else x for x in input_array]
    input_array = ["lamp_1" if x==25 else x for x in input_array]
    input_array = ["lamp_2" if x==26 else x for x in input_array]
    input_array = ["lamp_3" if x==27 else x for x in input_array]
    input_array = ["laptop_0" if x==28 else x for x in input_array]
    input_array = ["laptop_1" if x==29 else x for x in input_array]
    input_array = ["motor_0" if x==30 else x for x in input_array]
    input_array = ["motor_1" if x==31 else x for x in input_array]
    input_array = ["motor_2" if x==32 else x for x in input_array]
    input_array = ["motor_3" if x==33 else x for x in input_array]
    input_array = ["motor_4" if x==34 else x for x in input_array]
    input_array = ["motor_5" if x==35 else x for x in input_array]
    input_array = ["mug_0" if x==36 else x for x in input_array]
    input_array = ["mug_1" if x==37 else x for x in input_array]
    input_array = ["pistol_0" if x==38 else x for x in input_array]
    input_array = ["pistol_1" if x==39 else x for x in input_array]
    input_array = ["pistol_2" if x==40 else x for x in input_array]
    input_array = ["rocket_0" if x==41 else x for x in input_array]
    input_array = ["rocket_1" if x==42 else x for x in input_array]
    input_array = ["rocket_2" if x==43 else x for x in input_array]
    input_array = ["skateboard_0" if x==44 else x for x in input_array]
    input_array = ["skateboard_1" if x==45 else x for x in input_array]
    input_array = ["skateboard_2" if x==46 else x for x in input_array]
    input_array = ["table_0" if x==47 else x for x in input_array]
    input_array = ["table_1" if x==48 else x for x in input_array]
    input_array = ["table_2" if x==49 else x for x in input_array]
    
    # converting to a pandas series
    input_array = np.array(input_array).flatten()
    input_array = pd.Series(input_array)
    
    return input_array

# function to prepare chamfer distance (CD) input for the ShapeNet Part dataset
def prepare_cd_input_shapenet_part(hidden_layer_output, predictions,
                                   cloud_size = 1024, layer_dimension = 256,
                                   examples_per_class = 10):
    # getting object-wise hidden layer outputs
    airplane_hidden_layer_output = hidden_layer_output[0:examples_per_class]
    bag_hidden_layer_output = hidden_layer_output[examples_per_class:2*examples_per_class]
    cap_hidden_layer_output = hidden_layer_output[2*examples_per_class:3*examples_per_class]
    car_hidden_layer_output = hidden_layer_output[3*examples_per_class:4*examples_per_class]
    chair_hidden_layer_output = hidden_layer_output[4*examples_per_class:5*examples_per_class]
    earphone_hidden_layer_output = hidden_layer_output[5*examples_per_class:6*examples_per_class]
    guitar_hidden_layer_output = hidden_layer_output[6*examples_per_class:7*examples_per_class]
    knife_hidden_layer_output = hidden_layer_output[7*examples_per_class:8*examples_per_class]
    lamp_hidden_layer_output = hidden_layer_output[8*examples_per_class:9*examples_per_class]
    laptop_hidden_layer_output = hidden_layer_output[9*examples_per_class:10*examples_per_class]
    motorbike_hidden_layer_output = hidden_layer_output[10*examples_per_class:11*examples_per_class]
    mug_hidden_layer_output = hidden_layer_output[11*examples_per_class:12*examples_per_class]
    pistol_hidden_layer_output = hidden_layer_output[12*examples_per_class:13*examples_per_class]
    rocket_hidden_layer_output = hidden_layer_output[13*examples_per_class:14*examples_per_class]
    skateboard_hidden_layer_output = hidden_layer_output[14*examples_per_class:15*examples_per_class]
    table_hidden_layer_output = hidden_layer_output[15*examples_per_class:16*examples_per_class]
    # getting object-wise predictions
    airplane_predictions = predictions[0:examples_per_class]
    bag_predictions = predictions[examples_per_class:2*examples_per_class]
    cap_predictions = predictions[2*examples_per_class:3*examples_per_class]
    car_predictions = predictions[3*examples_per_class:4*examples_per_class]
    chair_predictions = predictions[4*examples_per_class:5*examples_per_class]
    earphone_predictions = predictions[5*examples_per_class:6*examples_per_class]
    guitar_predictions = predictions[6*examples_per_class:7*examples_per_class]
    knife_predictions = predictions[7*examples_per_class:8*examples_per_class]
    lamp_predictions = predictions[8*examples_per_class:9*examples_per_class]
    laptop_predictions = predictions[9*examples_per_class:10*examples_per_class]
    motorbike_predictions = predictions[10*examples_per_class:11*examples_per_class]
    mug_predictions = predictions[11*examples_per_class:12*examples_per_class]
    pistol_predictions = predictions[12*examples_per_class:13*examples_per_class]
    rocket_predictions = predictions[13*examples_per_class:14*examples_per_class]
    skateboard_predictions = predictions[14*examples_per_class:15*examples_per_class]
    table_predictions = predictions[15*examples_per_class:16*examples_per_class]
    
    # restructuring object-wise hidden layer outputs
    airplane_hidden_layer_output = np.resize(airplane_hidden_layer_output, (examples_per_class,cloud_size,layer_dimension))
    bag_hidden_layer_output = np.resize(bag_hidden_layer_output, (examples_per_class,cloud_size,layer_dimension))
    cap_hidden_layer_output = np.resize(cap_hidden_layer_output, (examples_per_class,cloud_size,layer_dimension))
    car_hidden_layer_output = np.resize(car_hidden_layer_output, (examples_per_class,cloud_size,layer_dimension))
    chair_hidden_layer_output = np.resize(chair_hidden_layer_output, (examples_per_class,cloud_size,layer_dimension))
    earphone_hidden_layer_output = np.resize(earphone_hidden_layer_output, (examples_per_class,cloud_size,layer_dimension))
    guitar_hidden_layer_output = np.resize(guitar_hidden_layer_output, (examples_per_class,cloud_size,layer_dimension))
    knife_hidden_layer_output = np.resize(knife_hidden_layer_output, (examples_per_class,cloud_size,layer_dimension))
    lamp_hidden_layer_output = np.resize(lamp_hidden_layer_output, (examples_per_class,cloud_size,layer_dimension))
    laptop_hidden_layer_output = np.resize(laptop_hidden_layer_output, (examples_per_class,cloud_size,layer_dimension))
    motorbike_hidden_layer_output = np.resize(motorbike_hidden_layer_output, (examples_per_class,cloud_size,layer_dimension))
    mug_hidden_layer_output = np.resize(mug_hidden_layer_output, (examples_per_class,cloud_size,layer_dimension))
    pistol_hidden_layer_output = np.resize(pistol_hidden_layer_output, (examples_per_class,cloud_size,layer_dimension))
    rocket_hidden_layer_output = np.resize(rocket_hidden_layer_output, (examples_per_class,cloud_size,layer_dimension))
    skateboard_hidden_layer_output = np.resize(skateboard_hidden_layer_output, (examples_per_class,cloud_size,layer_dimension))
    table_hidden_layer_output = np.resize(table_hidden_layer_output, (examples_per_class,cloud_size,layer_dimension))
    # restructuring object-wise predictions
    airplane_predictions = np.resize(airplane_predictions, (examples_per_class,cloud_size,50))
    bag_predictions = np.resize(bag_predictions, (examples_per_class,cloud_size,50))
    cap_predictions = np.resize(cap_predictions, (examples_per_class,cloud_size,50))
    car_predictions = np.resize(car_predictions, (examples_per_class,cloud_size,50))
    chair_predictions = np.resize(chair_predictions, (examples_per_class,cloud_size,50))
    earphone_predictions = np.resize(earphone_predictions, (examples_per_class,cloud_size,50))
    guitar_predictions = np.resize(guitar_predictions, (examples_per_class,cloud_size,50))
    knife_predictions = np.resize(knife_predictions, (examples_per_class,cloud_size,50))
    lamp_predictions = np.resize(lamp_predictions, (examples_per_class,cloud_size,50))
    laptop_predictions = np.resize(laptop_predictions, (examples_per_class,cloud_size,50))
    motorbike_predictions = np.resize(motorbike_predictions, (examples_per_class,cloud_size,50))
    mug_predictions = np.resize(mug_predictions, (examples_per_class,cloud_size,50))
    pistol_predictions = np.resize(pistol_predictions, (examples_per_class,cloud_size,50))
    rocket_predictions = np.resize(rocket_predictions, (examples_per_class,cloud_size,50))
    skateboard_predictions = np.resize(skateboard_predictions, (examples_per_class,cloud_size,50))
    table_predictions = np.resize(table_predictions, (examples_per_class,cloud_size,50))
    
    # dropping the one-hot encoding dimension for predictions
    airplane_predictions = th.from_numpy(airplane_predictions).max(dim=2)[1].detach().cpu().numpy()
    bag_predictions = th.from_numpy(bag_predictions).max(dim=2)[1].detach().cpu().numpy()
    cap_predictions = th.from_numpy(cap_predictions).max(dim=2)[1].detach().cpu().numpy()
    car_predictions = th.from_numpy(car_predictions).max(dim=2)[1].detach().cpu().numpy()
    chair_predictions = th.from_numpy(chair_predictions).max(dim=2)[1].detach().cpu().numpy()
    earphone_predictions = th.from_numpy(earphone_predictions).max(dim=2)[1].detach().cpu().numpy()
    guitar_predictions = th.from_numpy(guitar_predictions).max(dim=2)[1].detach().cpu().numpy()
    knife_predictions = th.from_numpy(knife_predictions).max(dim=2)[1].detach().cpu().numpy()
    lamp_predictions = th.from_numpy(lamp_predictions).max(dim=2)[1].detach().cpu().numpy()
    laptop_predictions = th.from_numpy(laptop_predictions).max(dim=2)[1].detach().cpu().numpy()
    motorbike_predictions = th.from_numpy(motorbike_predictions).max(dim=2)[1].detach().cpu().numpy()
    mug_predictions = th.from_numpy(mug_predictions).max(dim=2)[1].detach().cpu().numpy()
    pistol_predictions = th.from_numpy(pistol_predictions).max(dim=2)[1].detach().cpu().numpy()
    rocket_predictions = th.from_numpy(rocket_predictions).max(dim=2)[1].detach().cpu().numpy()
    skateboard_predictions = th.from_numpy(skateboard_predictions).max(dim=2)[1].detach().cpu().numpy()
    table_predictions = th.from_numpy(table_predictions).max(dim=2)[1].detach().cpu().numpy()
    
    # creating boolean masks per part and per example
    indices_airplane_part_0 = np.array(airplane_predictions == 0)
    indices_airplane_part_1 = np.array(airplane_predictions == 1)
    indices_airplane_part_2 = np.array(airplane_predictions == 2)
    indices_airplane_part_3 = np.array(airplane_predictions == 3)
    indices_bag_part_0 = np.array(bag_predictions == 4)
    indices_bag_part_1 = np.array(bag_predictions == 5)
    indices_cap_part_0 = np.array(cap_predictions == 6)
    indices_cap_part_1 = np.array(cap_predictions == 7)
    indices_car_part_0 = np.array(car_predictions == 8)
    indices_car_part_1 = np.array(car_predictions == 9)
    indices_car_part_2 = np.array(car_predictions == 10)
    indices_car_part_3 = np.array(car_predictions == 11)
    indices_chair_part_0 = np.array(chair_predictions == 12)
    indices_chair_part_1 = np.array(chair_predictions == 13)
    indices_chair_part_2 = np.array(chair_predictions == 14)
    indices_earphone_part_0 = np.array(earphone_predictions == 16)
    indices_earphone_part_1 = np.array(earphone_predictions == 17)
    indices_earphone_part_2 = np.array(earphone_predictions == 18)
    indices_guitar_part_0 = np.array(guitar_predictions == 19)
    indices_guitar_part_1 = np.array(guitar_predictions == 20)
    indices_guitar_part_2 = np.array(guitar_predictions == 21)
    indices_knife_part_0 = np.array(knife_predictions == 22)
    indices_knife_part_1 = np.array(knife_predictions == 23)
    indices_lamp_part_0 = np.array(lamp_predictions == 24)
    indices_lamp_part_1 = np.array(lamp_predictions == 25)
    indices_lamp_part_2 = np.array(lamp_predictions == 26)
    indices_lamp_part_3 = np.array(lamp_predictions == 27)
    indices_laptop_part_0 = np.array(laptop_predictions == 28)
    indices_laptop_part_1 = np.array(laptop_predictions == 29)
    indices_motorbike_part_0 = np.array(motorbike_predictions == 30)
    indices_motorbike_part_1 = np.array(motorbike_predictions == 31)
    indices_motorbike_part_2 = np.array(motorbike_predictions == 32)
    indices_motorbike_part_3 = np.array(motorbike_predictions == 33)
    indices_motorbike_part_4 = np.array(motorbike_predictions == 35)
    indices_mug_part_0 = np.array(mug_predictions == 36)
    indices_mug_part_1 = np.array(mug_predictions == 37)
    indices_pistol_part_0 = np.array(pistol_predictions == 38)
    indices_pistol_part_1 = np.array(pistol_predictions == 39)
    indices_pistol_part_2 = np.array(pistol_predictions == 40)
    indices_rocket_part_0 = np.array(rocket_predictions == 41)
    indices_rocket_part_1 = np.array(rocket_predictions == 42)
    indices_rocket_part_2 = np.array(rocket_predictions == 43)
    indices_skateboard_part_0 = np.array(skateboard_predictions == 44)
    indices_skateboard_part_1 = np.array(skateboard_predictions == 45)
    indices_skateboard_part_2 = np.array(skateboard_predictions == 46)
    indices_table_part_0 = np.array(table_predictions == 47)
    indices_table_part_1 = np.array(table_predictions == 48)
    indices_table_part_2 = np.array(table_predictions == 49)
    
    # applying part masks to get array subsets per example part
    # creating placeholders
    airplane_hidden_layer_output_part_0 = []
    airplane_hidden_layer_output_part_1 = []
    airplane_hidden_layer_output_part_2 = []
    airplane_hidden_layer_output_part_3 = []
    bag_hidden_layer_output_part_0 = []
    bag_hidden_layer_output_part_1 = []
    cap_hidden_layer_output_part_0 = []
    cap_hidden_layer_output_part_1 = []
    car_hidden_layer_output_part_0 = []
    car_hidden_layer_output_part_1 = []
    car_hidden_layer_output_part_2 = []
    car_hidden_layer_output_part_3 = []
    chair_hidden_layer_output_part_0 = []
    chair_hidden_layer_output_part_1 = []
    chair_hidden_layer_output_part_2 = []
    earphone_hidden_layer_output_part_0 = []
    earphone_hidden_layer_output_part_1 = []
    earphone_hidden_layer_output_part_2 = []
    guitar_hidden_layer_output_part_0 = []
    guitar_hidden_layer_output_part_1 = []
    guitar_hidden_layer_output_part_2 = []
    knife_hidden_layer_output_part_0 = []
    knife_hidden_layer_output_part_1 = []
    lamp_hidden_layer_output_part_0 = []
    lamp_hidden_layer_output_part_1 = []
    lamp_hidden_layer_output_part_2 = []
    lamp_hidden_layer_output_part_3 = []
    laptop_hidden_layer_output_part_0 = []
    laptop_hidden_layer_output_part_1 = []
    motorbike_hidden_layer_output_part_0 = []
    motorbike_hidden_layer_output_part_1 = []
    motorbike_hidden_layer_output_part_2 = []
    motorbike_hidden_layer_output_part_3 = []
    motorbike_hidden_layer_output_part_4 = []
    mug_hidden_layer_output_part_0 = []
    mug_hidden_layer_output_part_1 = []
    pistol_hidden_layer_output_part_0 = []
    pistol_hidden_layer_output_part_1 = []
    pistol_hidden_layer_output_part_2 = []
    rocket_hidden_layer_output_part_0 = []
    rocket_hidden_layer_output_part_1 = []
    rocket_hidden_layer_output_part_2 = []
    skateboard_hidden_layer_output_part_0 = []
    skateboard_hidden_layer_output_part_1 = []
    skateboard_hidden_layer_output_part_2 = []
    table_hidden_layer_output_part_0 = []
    table_hidden_layer_output_part_1 = []
    table_hidden_layer_output_part_2 = []
    
    # main loop for part subsetting per example
    for i in range(airplane_predictions.shape[0]):
        # subsetting by example parts
        inner_result_airplane_0 = airplane_hidden_layer_output[i][indices_airplane_part_0[i]]
        inner_result_airplane_1 = airplane_hidden_layer_output[i][indices_airplane_part_1[i]]
        inner_result_airplane_2 = airplane_hidden_layer_output[i][indices_airplane_part_2[i]]
        inner_result_airplane_3 = airplane_hidden_layer_output[i][indices_airplane_part_3[i]]
        inner_result_bag_0 = bag_hidden_layer_output[i][indices_bag_part_0[i]]
        inner_result_bag_1 = bag_hidden_layer_output[i][indices_bag_part_1[i]]
        inner_result_cap_0 = cap_hidden_layer_output[i][indices_cap_part_0[i]]
        inner_result_cap_1 = cap_hidden_layer_output[i][indices_cap_part_1[i]]
        inner_result_car_0 = car_hidden_layer_output[i][indices_car_part_0[i]]
        inner_result_car_1 = car_hidden_layer_output[i][indices_car_part_1[i]]
        inner_result_car_2 = car_hidden_layer_output[i][indices_car_part_2[i]]
        inner_result_car_3 = car_hidden_layer_output[i][indices_car_part_3[i]]
        inner_result_chair_0 = chair_hidden_layer_output[i][indices_chair_part_0[i]]
        inner_result_chair_1 = chair_hidden_layer_output[i][indices_chair_part_1[i]]
        inner_result_chair_2 = chair_hidden_layer_output[i][indices_chair_part_2[i]]
        inner_result_earphone_0 = earphone_hidden_layer_output[i][indices_earphone_part_0[i]]
        inner_result_earphone_1 = earphone_hidden_layer_output[i][indices_earphone_part_1[i]]
        inner_result_earphone_2 = earphone_hidden_layer_output[i][indices_earphone_part_2[i]]
        inner_result_guitar_0 = guitar_hidden_layer_output[i][indices_guitar_part_0[i]]
        inner_result_guitar_1 = guitar_hidden_layer_output[i][indices_guitar_part_1[i]]
        inner_result_guitar_2 = guitar_hidden_layer_output[i][indices_guitar_part_2[i]]
        inner_result_knife_0 = knife_hidden_layer_output[i][indices_knife_part_0[i]]
        inner_result_knife_1 = knife_hidden_layer_output[i][indices_knife_part_1[i]]
        inner_result_lamp_0 = lamp_hidden_layer_output[i][indices_lamp_part_0[i]]
        inner_result_lamp_1 = lamp_hidden_layer_output[i][indices_lamp_part_1[i]]
        inner_result_lamp_2 = lamp_hidden_layer_output[i][indices_lamp_part_2[i]]
        inner_result_lamp_3 = lamp_hidden_layer_output[i][indices_lamp_part_3[i]]
        inner_result_laptop_0 = laptop_hidden_layer_output[i][indices_laptop_part_0[i]]
        inner_result_laptop_1 = laptop_hidden_layer_output[i][indices_laptop_part_1[i]]
        inner_result_motorbike_0 = motorbike_hidden_layer_output[i][indices_motorbike_part_0[i]]
        inner_result_motorbike_1 = motorbike_hidden_layer_output[i][indices_motorbike_part_1[i]]
        inner_result_motorbike_2 = motorbike_hidden_layer_output[i][indices_motorbike_part_2[i]]
        inner_result_motorbike_3 = motorbike_hidden_layer_output[i][indices_motorbike_part_3[i]]
        inner_result_motorbike_4 = motorbike_hidden_layer_output[i][indices_motorbike_part_4[i]]
        inner_result_mug_0 = mug_hidden_layer_output[i][indices_mug_part_0[i]]
        inner_result_mug_1 = mug_hidden_layer_output[i][indices_mug_part_1[i]]
        inner_result_pistol_0 = pistol_hidden_layer_output[i][indices_pistol_part_0[i]]
        inner_result_pistol_1 = pistol_hidden_layer_output[i][indices_pistol_part_1[i]]
        inner_result_pistol_2 = pistol_hidden_layer_output[i][indices_pistol_part_2[i]]
        inner_result_rocket_0 = rocket_hidden_layer_output[i][indices_rocket_part_0[i]]
        inner_result_rocket_1 = rocket_hidden_layer_output[i][indices_rocket_part_1[i]]
        inner_result_rocket_2 = rocket_hidden_layer_output[i][indices_rocket_part_2[i]]
        inner_result_skateboard_0 = skateboard_hidden_layer_output[i][indices_skateboard_part_0[i]]
        inner_result_skateboard_1 = skateboard_hidden_layer_output[i][indices_skateboard_part_1[i]]
        inner_result_skateboard_2 = skateboard_hidden_layer_output[i][indices_skateboard_part_2[i]]
        inner_result_table_0 = table_hidden_layer_output[i][indices_table_part_0[i]]
        inner_result_table_1 = table_hidden_layer_output[i][indices_table_part_1[i]]
        inner_result_table_2 = table_hidden_layer_output[i][indices_table_part_2[i]]
        
        # appending to lists
        airplane_hidden_layer_output_part_0.append(inner_result_airplane_0)
        airplane_hidden_layer_output_part_1.append(inner_result_airplane_1)
        airplane_hidden_layer_output_part_2.append(inner_result_airplane_2)
        airplane_hidden_layer_output_part_3.append(inner_result_airplane_3)
        bag_hidden_layer_output_part_0.append(inner_result_bag_0)
        bag_hidden_layer_output_part_1.append(inner_result_bag_1)
        cap_hidden_layer_output_part_0.append(inner_result_cap_0)
        cap_hidden_layer_output_part_1.append(inner_result_cap_1)
        car_hidden_layer_output_part_0.append(inner_result_car_0)
        car_hidden_layer_output_part_1.append(inner_result_car_1)
        car_hidden_layer_output_part_2.append(inner_result_car_2)
        car_hidden_layer_output_part_3.append(inner_result_car_3)
        chair_hidden_layer_output_part_0.append(inner_result_chair_0)
        chair_hidden_layer_output_part_1.append(inner_result_chair_1)
        chair_hidden_layer_output_part_2.append(inner_result_chair_2)
        earphone_hidden_layer_output_part_0.append(inner_result_earphone_0)
        earphone_hidden_layer_output_part_1.append(inner_result_earphone_1)
        earphone_hidden_layer_output_part_2.append(inner_result_earphone_2)
        guitar_hidden_layer_output_part_0.append(inner_result_guitar_0)
        guitar_hidden_layer_output_part_1.append(inner_result_guitar_1)
        guitar_hidden_layer_output_part_2.append(inner_result_guitar_2)
        knife_hidden_layer_output_part_0.append(inner_result_knife_0)
        knife_hidden_layer_output_part_1.append(inner_result_knife_1)
        lamp_hidden_layer_output_part_0.append(inner_result_lamp_0)
        lamp_hidden_layer_output_part_1.append(inner_result_lamp_1)
        lamp_hidden_layer_output_part_2.append(inner_result_lamp_2)
        lamp_hidden_layer_output_part_3.append(inner_result_lamp_3)
        laptop_hidden_layer_output_part_0.append(inner_result_laptop_0)
        laptop_hidden_layer_output_part_1.append(inner_result_laptop_1)
        motorbike_hidden_layer_output_part_0.append(inner_result_motorbike_0)
        motorbike_hidden_layer_output_part_1.append(inner_result_motorbike_1)
        motorbike_hidden_layer_output_part_2.append(inner_result_motorbike_2)
        motorbike_hidden_layer_output_part_3.append(inner_result_motorbike_3)
        motorbike_hidden_layer_output_part_4.append(inner_result_motorbike_4)
        mug_hidden_layer_output_part_0.append(inner_result_mug_0)
        mug_hidden_layer_output_part_1.append(inner_result_mug_1)
        pistol_hidden_layer_output_part_0.append(inner_result_pistol_0)
        pistol_hidden_layer_output_part_1.append(inner_result_pistol_1)
        pistol_hidden_layer_output_part_2.append(inner_result_pistol_2)
        rocket_hidden_layer_output_part_0.append(inner_result_rocket_0)
        rocket_hidden_layer_output_part_1.append(inner_result_rocket_1)
        rocket_hidden_layer_output_part_2.append(inner_result_rocket_2)
        skateboard_hidden_layer_output_part_0.append(inner_result_skateboard_0)
        skateboard_hidden_layer_output_part_1.append(inner_result_skateboard_1)
        skateboard_hidden_layer_output_part_2.append(inner_result_skateboard_2)
        table_hidden_layer_output_part_0.append(inner_result_table_0)
        table_hidden_layer_output_part_1.append(inner_result_table_1)
        table_hidden_layer_output_part_2.append(inner_result_table_2)
        
    # stacking all example part sets together - input for Chamfer Distance
    chamfer_distance_input = np.vstack((airplane_hidden_layer_output_part_0,
                                        airplane_hidden_layer_output_part_1,
                                        airplane_hidden_layer_output_part_2,
                                        airplane_hidden_layer_output_part_3,
                                        bag_hidden_layer_output_part_0,
                                        bag_hidden_layer_output_part_1,
                                        cap_hidden_layer_output_part_0,
                                        cap_hidden_layer_output_part_1,
                                        car_hidden_layer_output_part_0,
                                        car_hidden_layer_output_part_1,
                                        car_hidden_layer_output_part_2,
                                        car_hidden_layer_output_part_3,
                                        chair_hidden_layer_output_part_0,
                                        chair_hidden_layer_output_part_1,
                                        chair_hidden_layer_output_part_2,
                                        earphone_hidden_layer_output_part_0,
                                        earphone_hidden_layer_output_part_1,
                                        earphone_hidden_layer_output_part_2,
                                        guitar_hidden_layer_output_part_0,
                                        guitar_hidden_layer_output_part_1,
                                        guitar_hidden_layer_output_part_2,
                                        knife_hidden_layer_output_part_0,
                                        knife_hidden_layer_output_part_1,
                                        lamp_hidden_layer_output_part_0,
                                        lamp_hidden_layer_output_part_1,
                                        lamp_hidden_layer_output_part_2,
                                        lamp_hidden_layer_output_part_3,
                                        laptop_hidden_layer_output_part_0,
                                        laptop_hidden_layer_output_part_1,
                                        motorbike_hidden_layer_output_part_0,
                                        motorbike_hidden_layer_output_part_1,
                                        motorbike_hidden_layer_output_part_2,
                                        motorbike_hidden_layer_output_part_3,
                                        motorbike_hidden_layer_output_part_4,
                                        mug_hidden_layer_output_part_0,
                                        mug_hidden_layer_output_part_1,
                                        pistol_hidden_layer_output_part_0,
                                        pistol_hidden_layer_output_part_1,
                                        pistol_hidden_layer_output_part_2,
                                        rocket_hidden_layer_output_part_0,
                                        rocket_hidden_layer_output_part_1,
                                        rocket_hidden_layer_output_part_2,
                                        skateboard_hidden_layer_output_part_0,
                                        skateboard_hidden_layer_output_part_1,
                                        skateboard_hidden_layer_output_part_2,
                                        table_hidden_layer_output_part_0,
                                        table_hidden_layer_output_part_1,
                                        table_hidden_layer_output_part_2
                                        ))
    
    # flattening the tensor
    cd_input = np.resize(chamfer_distance_input,
                         (chamfer_distance_input.shape[0]*chamfer_distance_input.shape[1],1)).flatten()
    
    return cd_input

# defining the color list for the PT-1 output on the ShapeNet Part subset
color_list_pt_1_conv9_subset_shapenet_part = []
# airplane
color_list_pt_1_conv9_subset_shapenet_part.append(['#CFD8DC'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#90A4AE'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#607D8B'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#455A64'] * 7)
# bag
color_list_pt_1_conv9_subset_shapenet_part.append(['#BDBDBD'] * 9)
color_list_pt_1_conv9_subset_shapenet_part.append(['#616161'] * 10)
# cap
color_list_pt_1_conv9_subset_shapenet_part.append(['#FF8A65'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#E64A19'] * 10)
# car
color_list_pt_1_conv9_subset_shapenet_part.append(['#BCAAA4'] * 6)
color_list_pt_1_conv9_subset_shapenet_part.append(['#8D6E63'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#6D4C41'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#4E342E'] * 10)
# chair
color_list_pt_1_conv9_subset_shapenet_part.append(['#FFE0B2'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#FFB74D'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#FB8C00'] * 10)
# earphone
color_list_pt_1_conv9_subset_shapenet_part.append(['#FFF9C4'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#FFF176'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#FDD835'] * 3)
# guitar
color_list_pt_1_conv9_subset_shapenet_part.append(['#DCE775'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#C0CA33'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#9E9D24'] * 10)
# knife
color_list_pt_1_conv9_subset_shapenet_part.append(['#4CAF50'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#1B5E20'] * 10)
# lamp
color_list_pt_1_conv9_subset_shapenet_part.append(['#80CBC4'] * 7)
color_list_pt_1_conv9_subset_shapenet_part.append(['#26A69A'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#00897B'] * 2)
color_list_pt_1_conv9_subset_shapenet_part.append(['#004D40'] * 10)
# laptop
color_list_pt_1_conv9_subset_shapenet_part.append(['#80DEEA'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#00BCD4'] * 10)
# motorbike
color_list_pt_1_conv9_subset_shapenet_part.append(['#BBDEFB'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#64B5F6'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#2196F3'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#1976D2'] * 8)
color_list_pt_1_conv9_subset_shapenet_part.append(['#0D47A1'] * 10)
# mug
color_list_pt_1_conv9_subset_shapenet_part.append(['#3F51B5'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#1A237E'] * 10)
# pistol
color_list_pt_1_conv9_subset_shapenet_part.append(['#B39DDB'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#7E57C2'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#512DA8'] * 10)
# rocket
color_list_pt_1_conv9_subset_shapenet_part.append(['#EF9A9A'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#EF5350'] * 8)
color_list_pt_1_conv9_subset_shapenet_part.append(['#C62828'] * 10)
# skateboard
color_list_pt_1_conv9_subset_shapenet_part.append(['#F8BBD0'] * 8)
color_list_pt_1_conv9_subset_shapenet_part.append(['#F06292'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#E91E63'] * 7)
# table
color_list_pt_1_conv9_subset_shapenet_part.append(['#E1BEE7'] * 10)
color_list_pt_1_conv9_subset_shapenet_part.append(['#BA68C8'] * 8)
color_list_pt_1_conv9_subset_shapenet_part.append(['#9C27B0'] * 2)
# post-processing
color_list_pt_1_conv9_subset_shapenet_part = [item for sublist in color_list_pt_1_conv9_subset_shapenet_part for item in sublist]
color_list_pt_1_conv9_subset_shapenet_part = pd.Series(color_list_pt_1_conv9_subset_shapenet_part)

