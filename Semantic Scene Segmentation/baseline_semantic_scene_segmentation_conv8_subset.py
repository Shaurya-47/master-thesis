# importing libraries
import torch as th
import numpy as np
import umap
import matplotlib.pyplot as plt
import pandas as pd
#%matplotlib inline

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

# defining the UMAP reducer
reducer = umap.UMAP(min_dist = 0.5, n_neighbors = 300, random_state = 1) # this is the original baseline
embedding = reducer.fit_transform(conv8_output)
#embedding.shape
# saving the embedding
np.save('./Results/semseg_conv8_baseline_embedding_2048_100.npy', embedding)

# pandas mapping approach
predictions = predictions.tolist()
predictions = ["ceiling" if x==0 else x for x in predictions]
predictions = ["floor" if x==1 else x for x in predictions]
predictions = ["wall" if x==2 else x for x in predictions]
predictions = ["beam" if x==3 else x for x in predictions]
predictions = ["column" if x==4 else x for x in predictions]
predictions = ["window" if x==5 else x for x in predictions]
predictions = ["door" if x==6 else x for x in predictions]
predictions = ["table" if x==7 else x for x in predictions]
predictions = ["chair" if x==8 else x for x in predictions]
predictions = ["sofa" if x==9 else x for x in predictions]
predictions = ["bookcase" if x==10 else x for x in predictions]
predictions = ["board" if x==11 else x for x in predictions]
predictions = ["clutter" if x==12 else x for x in predictions]

predictions = np.array(predictions).flatten()
predictions = pd.Series(predictions)
predictions.shape
#print(predictions.unique())

# NA detection
#predictions[predictions.isnull().any(0)]

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
    
colormap_applied = predictions.map(colormap)
#print(colormap_applied.unique())
#colormap_inspect[colormap_inspect.isnull().any(0)] # NA detection

# plot projection
plt.figure(figsize=(8,6), dpi=1000)
plt.scatter(embedding[:,0], embedding[:,1], s=0.05, c=colormap_applied)
plt.title('UMAP baseline projection (using predictions) \n on 100 samples (cloud size 1024)', fontsize=24)
