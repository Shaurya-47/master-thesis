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
module_path = os.path.abspath(os.path.join('.\GitHub\master-thesis\semantic_scene_segmentation'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.utils_sem_seg import data_loader_sem_seg, prepare_cd_input_s3dis, construct_cd_matrix, class_labeler, projection, visualize_projection, color_list_pt_1_conv8_subset_s3dis   

# execution
if __name__ == '__main__':
    conv8_hidden_output_subset, preds_subset, labels_subset = data_loader_sem_seg()
    # processing and transforming the data into an input form for the CD matrix
    cd_input_conv9 = prepare_cd_input_s3dis(conv8_hidden_output_subset, preds_subset)
    # compute the aggregated graph representing the high dimensional data
    cd_matrix_conv9 = construct_cd_matrix(cd_input_conv9)
    # obtaining UMAP embedding
    embedding = projection(cd_matrix_conv9, mindist = 1, neighbors = 200)
    # preprocessing prediction array for visualization
    preds_subset_strings = class_labeler(preds_subset)
    # visualizing the 2D embedding (colored by predictions)
    visualize_projection(embedding, color_list_pt_1_conv8_subset_s3dis,
                         title = 'UMAP PT-1 projection (colored via predictions) \n on a test subset of 100 samples (cloud size = 2048)')