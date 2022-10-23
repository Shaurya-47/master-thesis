# importing libraries
import torch as th
import numpy as np
import numpy as np
import umap
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
#%matplotlib inline

# importing utils functions
module_path = os.path.abspath(os.path.join('.\GitHub\master-thesis\part_segmentation'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.utils_part_seg import data_loader_part_seg, class_labeler, projection, visualize_projection, colormap
    
# execution
if __name__ == '__main__':
    # loading data
    conv9_hidden_output_subset, preds_subset, part_labels_subset = data_loader_part_seg()
    # obtaining UMAP embedding
    embedding = projection(conv9_hidden_output_subset)
    # preprocessing prediction array for visualization
    preds_subset_strings = class_labeler(preds_subset)
    # visualizing the 2D embedding (colored by predictions)
    visualize_projection(embedding, preds_subset_strings, colormap,
                         title = 'UMAP baseline projection (colored via predictions) \n on a test subset of 160 samples (cloud size = 1024)')