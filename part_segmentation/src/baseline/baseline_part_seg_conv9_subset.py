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

# importing utils file
module_path = os.path.abspath(os.path.join('.\GitHub\master-thesis\part_segmentation'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.utils_part_seg import class_labeler, projection, visualize_projection, colormap
    
# importing data
conv9_hidden_output_subset = th.load('./data/conv9_hidden_output_subset.pt')
preds_subset = th.load('./data/preds_subset.pt')
part_labels_subset = th.load('./data/part_labels_subset.pt')

# execution
if __name__ == '__main__':
    # obtaining UMAP embedding
    embedding = projection(conv9_hidden_output_subset)
    # preprocessing prediction array for visualization
    preds_subset_strings = class_labeler(preds_subset)
    # visualizing the 2D embedding (colored by predictions)
    visualize_projection(embedding, preds_subset_strings, colormap)
    
