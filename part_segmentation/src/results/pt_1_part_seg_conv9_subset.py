# library imports
import torch as th
import numpy as np
import umap
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from pytorch3d.loss import chamfer_distance as cd

# importing utils functions
module_path = os.path.abspath(os.path.join('.\GitHub\master-thesis\part_segmentation'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.utils_part_seg import data_loader_part_seg, prepare_cd_input_shapenet_part, construct_cd_matrix, class_labeler, projection, visualize_projection, color_list_pt_1_conv9_subset_shapenet_part     

# execution
if __name__ == '__main__':
    conv9_hidden_output_subset, preds_subset, part_labels_subset = data_loader_part_seg()
    # processing and transforming the data into an input form for the CD matrix
    cd_input_conv9 = prepare_cd_input_shapenet_part(conv9_hidden_output_subset, preds_subset)
    # compute the aggregated graph representing the high dimensional data
    cd_matrix_conv9 = construct_cd_matrix(cd_input_conv9)
    # obtaining UMAP embedding
    embedding = projection(cd_matrix_conv9, mindist = 1, neighbors = 190)
    # preprocessing prediction array for visualization
    preds_subset_strings = class_labeler(preds_subset)
    # visualizing the 2D embedding (colored by predictions)
    visualize_projection(embedding, color_list_pt_1_conv9_subset_shapenet_part,
                         title = 'UMAP PT-1 projection (colored via predictions) \n on a test subset of 160 samples (cloud size = 1024)')