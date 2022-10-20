# importing libraries
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt

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

# defined color map for part segmentation
colormap = {"airplane_0": "#CFD8DC", "airplane_1": "#90A4AE", "airplane_2": "#607D8B", "airplane_3": "#455A64",
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

# function to obtain a UMAP 2D embedding from high dimensional data
def projection(hd_data, min_dist = 0.5, n_neighbors = 300, rs = 1):
    # defining the UMAP reducer
    reducer = umap.UMAP(min_dist = 0.5, n_neighbors = 300, random_state = 1) # this is the original baseline
    # applying UMAP to reduce the dimensionality to 2
    embedding = reducer.fit_transform(hd_data)
    
    return embedding

# function to visualize the UMAP embedding
def visualize_projection(embedding_input, string_labels, color_map):
    plt.figure(figsize=(8,6), dpi=1000)
    plt.scatter(embedding_input[:,0], embedding_input[:,1], s=0.05,
                c=string_labels.map(colormap))
    plt.title('UMAP baseline projection (colored via predictions) \n on a test subset of 160 samples (cloud size = 1024)',
              fontsize=24)
    plt.show()