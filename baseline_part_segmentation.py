# importing libraries
import torch as th
import numpy as np
import umap
#import matplotlib.pyplot as plt
#import seaborn as sns
#import pandas as pd
#from sklearn.preprocessing import StandardScaler
#%matplotlib inline

# importing data - default numpy array format
airplane_test_subset_conv9_hidden_output = th.load('./Data/partseg_full_data_conv9/airplane_test_subset_conv9_hidden_output_big_dataset.pt')
bag_test_subset_conv9_hidden_output = th.load('./Data/partseg_full_data_conv9/bag_test_subset_conv9_hidden_output_big_dataset.pt')
cap_test_subset_conv9_hidden_output = th.load('./Data/partseg_full_data_conv9/cap_test_subset_conv9_hidden_output_big_dataset.pt')
car_test_subset_conv9_hidden_output = th.load('./Data/partseg_full_data_conv9/car_test_subset_conv9_hidden_output_big_dataset.pt')
chair_test_subset_conv9_hidden_output = th.load('./Data/partseg_full_data_conv9/chair_test_subset_conv9_hidden_output_big_dataset.pt')
earphone_test_subset_conv9_hidden_output = th.load('./Data/partseg_full_data_conv9/earphone_test_subset_conv9_hidden_output_big_dataset.pt')
guitar_test_subset_conv9_hidden_output = th.load('./Data/partseg_full_data_conv9/guitar_test_subset_conv9_hidden_output_big_dataset.pt')
knife_test_subset_conv9_hidden_output = th.load('./Data/partseg_full_data_conv9/knife_test_subset_conv9_hidden_output_big_dataset.pt')
lamp_test_subset_conv9_hidden_output = th.load('./Data/partseg_full_data_conv9/lamp_test_subset_conv9_hidden_output_big_dataset.pt')
laptop_test_subset_conv9_hidden_output = th.load('./Data/partseg_full_data_conv9/laptop_test_subset_conv9_hidden_output_big_dataset.pt')
motorbike_test_subset_conv9_hidden_output = th.load('./Data/partseg_full_data_conv9/motorbike_test_subset_conv9_hidden_output_big_dataset.pt')
mug_test_subset_conv9_hidden_output = th.load('./Data/partseg_full_data_conv9/mug_test_subset_conv9_hidden_output_big_dataset.pt')
pistol_test_subset_conv9_hidden_output = th.load('./Data/partseg_full_data_conv9/pistol_test_subset_conv9_hidden_output_big_dataset.pt')
rocket_test_subset_conv9_hidden_output = th.load('./Data/partseg_full_data_conv9/rocket_test_subset_conv9_hidden_output_big_dataset.pt')
skateboard_test_subset_conv9_hidden_output = th.load('./Data/partseg_full_data_conv9/skateboard_test_subset_conv9_hidden_output_big_dataset.pt')
table_test_subset_conv9_hidden_output = th.load('./Data/partseg_full_data_conv9/table_test_subset_conv9_hidden_output_big_dataset.pt')

airplane_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/airplane_test_subset_part_labels_big_dataset.pt')
bag_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/bag_test_subset_part_labels_big_dataset.pt')
cap_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/cap_test_subset_part_labels_big_dataset.pt')
car_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/car_test_subset_part_labels_big_dataset.pt')
chair_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/chair_test_subset_part_labels_big_dataset.pt')
earphone_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/earphone_test_subset_part_labels_big_dataset.pt')
guitar_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/guitar_test_subset_part_labels_big_dataset.pt')
knife_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/knife_test_subset_part_labels_big_dataset.pt')
lamp_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/lamp_test_subset_part_labels_big_dataset.pt')
laptop_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/laptop_test_subset_part_labels_big_dataset.pt')
motorbike_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/motorbike_test_subset_part_labels_big_dataset.pt')
mug_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/mug_test_subset_part_labels_big_dataset.pt')
pistol_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/pistol_test_subset_part_labels_big_dataset.pt')
rocket_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/rocket_test_subset_part_labels_big_dataset.pt')
skateboard_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/skateboard_test_subset_part_labels_big_dataset.pt')
table_test_subset_part_labels = th.load('./Data/partseg_full_data_conv9/table_test_subset_part_labels_big_dataset.pt')

airplane_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/airplane_test_subset_predictions_big_dataset.pt')
bag_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/bag_test_subset_predictions_big_dataset.pt')
cap_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/cap_test_subset_predictions_big_dataset.pt')
car_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/car_test_subset_predictions_big_dataset.pt')
chair_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/chair_test_subset_predictions_big_dataset.pt')
earphone_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/earphone_test_subset_predictions_big_dataset.pt')
guitar_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/guitar_test_subset_predictions_big_dataset.pt')
knife_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/knife_test_subset_predictions_big_dataset.pt')
lamp_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/lamp_test_subset_predictions_big_dataset.pt')
laptop_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/laptop_test_subset_predictions_big_dataset.pt')
motorbike_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/motorbike_test_subset_predictions_big_dataset.pt')
mug_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/mug_test_subset_predictions_big_dataset.pt')
pistol_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/pistol_test_subset_predictions_big_dataset.pt')
rocket_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/rocket_test_subset_predictions_big_dataset.pt')
skateboard_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/skateboard_test_subset_predictions_big_dataset.pt')
table_test_subset_predictions = th.load('./Data/partseg_full_data_conv9/table_test_subset_predictions_big_dataset.pt')

# defining the UMAP reducer
reducer = umap.UMAP(min_dist = 0.5, n_neighbors = 300, random_state = 1) # this is the original baseline


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

# applying UMAP to reduce the dimensionality to 2
embedding = reducer.fit_transform(conv9_hidden_output_subset)
print(embedding.shape)

# saving the conv9 embedding locally so I can load it directly next time
np.save('./Results/conv9_full_data_baseline_UMAP_embedding_hp_0.5_300_rs_1.npy', embedding)
