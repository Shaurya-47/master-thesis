# library imports
import torch as th
import numpy as np
import umap
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from pytorch3d.loss import chamfer_distance as cd

# importing utils file
module_path = os.path.abspath(os.path.join('.\GitHub\master-thesis\part_segmentation'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.utils_part_seg import data_loader_part_seg, prepare_cd_input_shapenet_part, construct_cd_matrix, class_labeler, projection, visualize_projection, colormap


#################################### import this as a variable from utlis

colormap = []
# airplane
colormap.append(['#CFD8DC'] * 10)
colormap.append(['#90A4AE'] * 10)
colormap.append(['#607D8B'] * 10)
colormap.append(['#455A64'] * 7)

# bag
colormap.append(['#BDBDBD'] * 9)
colormap.append(['#616161'] * 10)

# cap
colormap.append(['#FF8A65'] * 10)
colormap.append(['#E64A19'] * 10)

# car
colormap.append(['#BCAAA4'] * 6)
colormap.append(['#8D6E63'] * 10)
colormap.append(['#6D4C41'] * 10)
colormap.append(['#4E342E'] * 10)

# chair
colormap.append(['#FFE0B2'] * 10)
colormap.append(['#FFB74D'] * 10)
colormap.append(['#FB8C00'] * 10)

# earphone
colormap.append(['#FFF9C4'] * 10)
colormap.append(['#FFF176'] * 10)
colormap.append(['#FDD835'] * 3)

# guitar
colormap.append(['#DCE775'] * 10)
colormap.append(['#C0CA33'] * 10)
colormap.append(['#9E9D24'] * 10)

# knife
colormap.append(['#4CAF50'] * 10)
colormap.append(['#1B5E20'] * 10)

# lamp
colormap.append(['#80CBC4'] * 7)
colormap.append(['#26A69A'] * 10)
colormap.append(['#00897B'] * 2)
colormap.append(['#004D40'] * 10)

# laptop
colormap.append(['#80DEEA'] * 10)
colormap.append(['#00BCD4'] * 10)

# motorbike
colormap.append(['#BBDEFB'] * 10)
colormap.append(['#64B5F6'] * 10)
colormap.append(['#2196F3'] * 10)
colormap.append(['#1976D2'] * 8)
colormap.append(['#0D47A1'] * 10)

# mug
colormap.append(['#3F51B5'] * 10)
colormap.append(['#1A237E'] * 10)

# pistol
colormap.append(['#B39DDB'] * 10)
colormap.append(['#7E57C2'] * 10)
colormap.append(['#512DA8'] * 10)

# rocket
colormap.append(['#EF9A9A'] * 10)
colormap.append(['#EF5350'] * 8)
colormap.append(['#C62828'] * 10)

# skateboard
colormap.append(['#F8BBD0'] * 8)
colormap.append(['#F06292'] * 10)
colormap.append(['#E91E63'] * 7)

# table
colormap.append(['#E1BEE7'] * 10)
colormap.append(['#BA68C8'] * 8)
colormap.append(['#9C27B0'] * 2)

# BLUE GREY: airplane
# GREY: bag
# D.ORANGE: cap
# BROWN: car
# ORANGE: chair
# YELLOW: earphone
# LIME: guitar
# GREEN: knife
# TEAL: lamp
# CYAN: laptop
# BLUE: motorbike
# INDIGO: mug
# D.PURPLE: pistol
# RED: rocket
# PINK: skateboard
# PURPLE: table

colormap = [item for sublist in colormap for item in sublist]
colormap = pd.Series(colormap)
colormap.unique()

# applying UMAP
reducer = umap.UMAP(min_dist = 1, n_neighbors = 190, metric = 'precomputed',
                    random_state = 1)
embedding = reducer.fit_transform(cdm_baseline_subset)
embedding.shape       

# plotting
plt.figure(figsize=(8,6), dpi=800)
plt.scatter(embedding[:,0], embedding[:,1], s=7, marker = 'v', c=colormap)
plt.title('UMAP PT-1 projection (using predictions) \n on 10 samples (size 1024) from all 16 categories', fontsize=24)

# execution
if __name__ == '__main__':
    conv9_hidden_output_subset, preds_subset, part_labels_subset = data_loader_part_seg()
    # processing and transforming the data into an input form for the CD matrix
    cd_input_conv9 = prepare_cd_input_shapenet_part(conv9_hidden_output_subset, preds_subset)
    # compute the aggregated graph representing the high dimensional data
    cd_matrix_conv9 = construct_cd_matrix(cd_input_conv9)
    
    # use the aggregated graph to compute the PT-2 UMAP embedding
    embedding_pt_2 = umap_embedding_pt_2(graph_agg)
    # preprocessing prediction array for visualization
    preds_subset_strings = class_labeler(graph_part_reference)
    # visualizing the PT-2 2D embedding (colored by predictions)
    visualize_projection(embedding_pt_2, preds_subset_strings, colormap,
                         title = 'UMAP PT-2 projection (colored via predictions) \n on a test subset of 160 samples (cloud size = 1024)')