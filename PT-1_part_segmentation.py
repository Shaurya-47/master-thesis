# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 16:55:59 2022

@author: Shaurya
"""

# Chamfer distance calculation on a subset of parents and their parts

# library imports
import torch as th
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#from sklearn.preprocessing import StandardScaler
#from sklearn.neighbors import NearestNeighbors

# importing data
airplane_test_subset_conv9_hidden_output = th.load('airplane_test_subset_conv9_hidden_output.pt')
bag_test_subset_conv9_hidden_output = th.load('bag_test_subset_conv9_hidden_output.pt')
cap_test_subset_conv9_hidden_output = th.load('cap_test_subset_conv9_hidden_output.pt')
car_test_subset_conv9_hidden_output = th.load('car_test_subset_conv9_hidden_output.pt')
chair_test_subset_conv9_hidden_output = th.load('chair_test_subset_conv9_hidden_output.pt')
earphone_test_subset_conv9_hidden_output = th.load('earphone_test_subset_conv9_hidden_output.pt')
guitar_test_subset_conv9_hidden_output = th.load('guitar_test_subset_conv9_hidden_output.pt')
knife_test_subset_conv9_hidden_output = th.load('knife_test_subset_conv9_hidden_output.pt')
lamp_test_subset_conv9_hidden_output = th.load('lamp_test_subset_conv9_hidden_output.pt')
laptop_test_subset_conv9_hidden_output = th.load('laptop_test_subset_conv9_hidden_output.pt')
motorbike_test_subset_conv9_hidden_output = th.load('motorbike_test_subset_conv9_hidden_output.pt')
mug_test_subset_conv9_hidden_output = th.load('mug_test_subset_conv9_hidden_output.pt')
pistol_test_subset_conv9_hidden_output = th.load('pistol_test_subset_conv9_hidden_output.pt')
rocket_test_subset_conv9_hidden_output = th.load('rocket_test_subset_conv9_hidden_output.pt')
skateboard_test_subset_conv9_hidden_output = th.load('skateboard_test_subset_conv9_hidden_output.pt')
table_test_subset_conv9_hidden_output = th.load('table_test_subset_conv9_hidden_output.pt')

airplane_test_subset_part_labels = th.load('airplane_test_subset_part_labels.pt')
bag_test_subset_part_labels = th.load('bag_test_subset_part_labels.pt')
cap_test_subset_part_labels = th.load('cap_test_subset_part_labels.pt')
car_test_subset_part_labels = th.load('car_test_subset_part_labels.pt')
chair_test_subset_part_labels = th.load('chair_test_subset_part_labels.pt')
earphone_test_subset_part_labels = th.load('earphone_test_subset_part_labels.pt')
guitar_test_subset_part_labels = th.load('guitar_test_subset_part_labels.pt')
knife_test_subset_part_labels = th.load('knife_test_subset_part_labels.pt')
lamp_test_subset_part_labels = th.load('lamp_test_subset_part_labels.pt')
laptop_test_subset_part_labels = th.load('laptop_test_subset_part_labels.pt')
motorbike_test_subset_part_labels = th.load('motorbike_test_subset_part_labels.pt')
mug_test_subset_part_labels = th.load('mug_test_subset_part_labels.pt')
pistol_test_subset_part_labels = th.load('pistol_test_subset_part_labels.pt')
rocket_test_subset_part_labels = th.load('rocket_test_subset_part_labels.pt')
skateboard_test_subset_part_labels = th.load('skateboard_test_subset_part_labels.pt')
table_test_subset_part_labels = th.load('table_test_subset_part_labels.pt')

airplane_test_subset_predictions = th.load('airplane_test_subset_predictions.pt')
bag_test_subset_predictions = th.load('bag_test_subset_predictions.pt')
cap_test_subset_predictions = th.load('cap_test_subset_predictions.pt')
car_test_subset_predictions = th.load('car_test_subset_predictions.pt')
chair_test_subset_predictions = th.load('chair_test_subset_predictions.pt')
earphone_test_subset_predictions = th.load('earphone_test_subset_predictions.pt')
guitar_test_subset_predictions = th.load('guitar_test_subset_predictions.pt')
knife_test_subset_predictions = th.load('knife_test_subset_predictions.pt')
lamp_test_subset_predictions = th.load('lamp_test_subset_predictions.pt')
laptop_test_subset_predictions = th.load('laptop_test_subset_predictions.pt')
motorbike_test_subset_predictions = th.load('motorbike_test_subset_predictions.pt')
mug_test_subset_predictions = th.load('mug_test_subset_predictions.pt')
pistol_test_subset_predictions = th.load('pistol_test_subset_predictions.pt')
rocket_test_subset_predictions = th.load('rocket_test_subset_predictions.pt')
skateboard_test_subset_predictions = th.load('skateboard_test_subset_predictions.pt')
table_test_subset_predictions = th.load('table_test_subset_predictions.pt')

# Chamfer distance function
# source: taken from PyTorch3D


# restructuring tensors as batch_size x num_points x dimension
airplane_conv9_hidden_output = np.moveaxis(airplane_test_subset_conv9_hidden_output, 2, 1)
bag_conv9_hidden_output = np.moveaxis(bag_test_subset_conv9_hidden_output, 2, 1)
cap_conv9_hidden_output = np.moveaxis(cap_test_subset_conv9_hidden_output, 2, 1)
car_conv9_hidden_output = np.moveaxis(car_test_subset_conv9_hidden_output, 2, 1)
chair_conv9_hidden_output = np.moveaxis(chair_test_subset_conv9_hidden_output, 2, 1)
earphone_conv9_hidden_output = np.moveaxis(earphone_test_subset_conv9_hidden_output, 2, 1)
guitar_conv9_hidden_output = np.moveaxis(guitar_test_subset_conv9_hidden_output, 2, 1)
knife_conv9_hidden_output = np.moveaxis(knife_test_subset_conv9_hidden_output, 2, 1)
lamp_conv9_hidden_output = np.moveaxis(lamp_test_subset_conv9_hidden_output, 2, 1)
laptop_conv9_hidden_output = np.moveaxis(laptop_test_subset_conv9_hidden_output, 2, 1)
motorbike_conv9_hidden_output = np.moveaxis(motorbike_test_subset_conv9_hidden_output, 2, 1)
mug_conv9_hidden_output = np.moveaxis(mug_test_subset_conv9_hidden_output, 2, 1)
pistol_conv9_hidden_output = np.moveaxis(pistol_test_subset_conv9_hidden_output, 2, 1)
rocket_conv9_hidden_output = np.moveaxis(rocket_test_subset_conv9_hidden_output, 2, 1)
skateboard_conv9_hidden_output = np.moveaxis(skateboard_test_subset_conv9_hidden_output, 2, 1)
table_conv9_hidden_output = np.moveaxis(table_test_subset_conv9_hidden_output, 2, 1)

# Applying Chamfer distance between all parent parts 
conv9_hidden_output_subset = np.vstack((airplane_conv9_hidden_output,
                                        bag_conv9_hidden_output,
                                        cap_conv9_hidden_output,
                                        car_conv9_hidden_output,
                                        chair_conv9_hidden_output,
                                        earphone_conv9_hidden_output,
                                        guitar_conv9_hidden_output,
                                        knife_conv9_hidden_output,
                                        lamp_conv9_hidden_output,
                                        laptop_conv9_hidden_output,
                                        motorbike_conv9_hidden_output,
                                        mug_conv9_hidden_output,
                                        pistol_conv9_hidden_output,
                                        rocket_conv9_hidden_output,
                                        skateboard_conv9_hidden_output,
                                        table_conv9_hidden_output))

# part_labels_subset = np.vstack((airplane_test_subset_part_labels, 
#                                 car_test_subset_part_labels,
#                                 guitar_test_subset_part_labels,
#                                 ))

predictions_subset = np.vstack((airplane_test_subset_predictions,
                                bag_test_subset_predictions,
                                cap_test_subset_predictions,
                                car_test_subset_predictions,
                                chair_test_subset_predictions,
                                earphone_test_subset_predictions,
                                guitar_test_subset_predictions,
                                knife_test_subset_predictions,
                                lamp_test_subset_predictions,
                                laptop_test_subset_predictions,
                                motorbike_test_subset_predictions,
                                mug_test_subset_predictions,
                                pistol_test_subset_predictions,
                                rocket_test_subset_predictions,
                                skateboard_test_subset_predictions,
                                table_test_subset_predictions
                                ))

predictions_subset = th.from_numpy(predictions_subset)
predictions_subset = predictions_subset.permute(0, 2, 1).contiguous()
predictions_subset = predictions_subset.max(dim=2)[1]
predictions_subset = predictions_subset.detach().cpu().numpy()

part_coherence_check = []
for i in range(predictions_subset.shape[0]):
    result = np.unique(predictions_subset[i])
    part_coherence_check.append(result)
    
# average:
count = []
for i in part_coherence_check:
    count.append(i.shape[0])
    
np.mean(count)

predictions_subset = np.resize(predictions_subset, (163840,1)).flatten()
np.unique(predictions_subset)
np.unique(predictions_subset).shape # 48/50 as a chair part and a motorcycle part are never predicted

# reshaping the array - batch_size will now be subdivided into parts

# object (parts): overall 48 parts (2 missing) and 48x10 = 480 entries
#np.unique(part_labels_subset)




# need predictions for each object separately
airplane_test_subset_predictions = th.from_numpy(airplane_test_subset_predictions)
airplane_test_subset_predictions = airplane_test_subset_predictions.permute(0, 2, 1).contiguous()
airplane_test_subset_predictions = airplane_test_subset_predictions.max(dim=2)[1]
airplane_test_subset_predictions = airplane_test_subset_predictions.detach().cpu().numpy()

bag_test_subset_predictions = th.from_numpy(bag_test_subset_predictions)
bag_test_subset_predictions = bag_test_subset_predictions.permute(0, 2, 1).contiguous()
bag_test_subset_predictions = bag_test_subset_predictions.max(dim=2)[1]
bag_test_subset_predictions = bag_test_subset_predictions.detach().cpu().numpy()

cap_test_subset_predictions = th.from_numpy(cap_test_subset_predictions)
cap_test_subset_predictions = cap_test_subset_predictions.permute(0, 2, 1).contiguous()
cap_test_subset_predictions = cap_test_subset_predictions.max(dim=2)[1]
cap_test_subset_predictions = cap_test_subset_predictions.detach().cpu().numpy()

car_test_subset_predictions = th.from_numpy(car_test_subset_predictions)
car_test_subset_predictions = car_test_subset_predictions.permute(0, 2, 1).contiguous()
car_test_subset_predictions = car_test_subset_predictions.max(dim=2)[1]
car_test_subset_predictions = car_test_subset_predictions.detach().cpu().numpy()

chair_test_subset_predictions = th.from_numpy(chair_test_subset_predictions)
chair_test_subset_predictions = chair_test_subset_predictions.permute(0, 2, 1).contiguous()
chair_test_subset_predictions = chair_test_subset_predictions.max(dim=2)[1]
chair_test_subset_predictions = chair_test_subset_predictions.detach().cpu().numpy()

earphone_test_subset_predictions = th.from_numpy(earphone_test_subset_predictions)
earphone_test_subset_predictions = earphone_test_subset_predictions.permute(0, 2, 1).contiguous()
earphone_test_subset_predictions = earphone_test_subset_predictions.max(dim=2)[1]
earphone_test_subset_predictions = earphone_test_subset_predictions.detach().cpu().numpy()

guitar_test_subset_predictions = th.from_numpy(guitar_test_subset_predictions)
guitar_test_subset_predictions = guitar_test_subset_predictions.permute(0, 2, 1).contiguous()
guitar_test_subset_predictions = guitar_test_subset_predictions.max(dim=2)[1]
guitar_test_subset_predictions = guitar_test_subset_predictions.detach().cpu().numpy()

knife_test_subset_predictions = th.from_numpy(knife_test_subset_predictions)
knife_test_subset_predictions = knife_test_subset_predictions.permute(0, 2, 1).contiguous()
knife_test_subset_predictions = knife_test_subset_predictions.max(dim=2)[1]
knife_test_subset_predictions = knife_test_subset_predictions.detach().cpu().numpy()

lamp_test_subset_predictions = th.from_numpy(lamp_test_subset_predictions)
lamp_test_subset_predictions = lamp_test_subset_predictions.permute(0, 2, 1).contiguous()
lamp_test_subset_predictions = lamp_test_subset_predictions.max(dim=2)[1]
lamp_test_subset_predictions = lamp_test_subset_predictions.detach().cpu().numpy()

laptop_test_subset_predictions = th.from_numpy(laptop_test_subset_predictions)
laptop_test_subset_predictions = laptop_test_subset_predictions.permute(0, 2, 1).contiguous()
laptop_test_subset_predictions = laptop_test_subset_predictions.max(dim=2)[1]
laptop_test_subset_predictions = laptop_test_subset_predictions.detach().cpu().numpy()

motorbike_test_subset_predictions = th.from_numpy(motorbike_test_subset_predictions)
motorbike_test_subset_predictions = motorbike_test_subset_predictions.permute(0, 2, 1).contiguous()
motorbike_test_subset_predictions = motorbike_test_subset_predictions.max(dim=2)[1]
motorbike_test_subset_predictions = motorbike_test_subset_predictions.detach().cpu().numpy()

mug_test_subset_predictions = th.from_numpy(mug_test_subset_predictions)
mug_test_subset_predictions = mug_test_subset_predictions.permute(0, 2, 1).contiguous()
mug_test_subset_predictions = mug_test_subset_predictions.max(dim=2)[1]
mug_test_subset_predictions = mug_test_subset_predictions.detach().cpu().numpy()

pistol_test_subset_predictions = th.from_numpy(pistol_test_subset_predictions)
pistol_test_subset_predictions = pistol_test_subset_predictions.permute(0, 2, 1).contiguous()
pistol_test_subset_predictions = pistol_test_subset_predictions.max(dim=2)[1]
pistol_test_subset_predictions = pistol_test_subset_predictions.detach().cpu().numpy()

rocket_test_subset_predictions = th.from_numpy(rocket_test_subset_predictions)
rocket_test_subset_predictions = rocket_test_subset_predictions.permute(0, 2, 1).contiguous()
rocket_test_subset_predictions = rocket_test_subset_predictions.max(dim=2)[1]
rocket_test_subset_predictions = rocket_test_subset_predictions.detach().cpu().numpy()

skateboard_test_subset_predictions = th.from_numpy(skateboard_test_subset_predictions)
skateboard_test_subset_predictions = skateboard_test_subset_predictions.permute(0, 2, 1).contiguous()
skateboard_test_subset_predictions = skateboard_test_subset_predictions.max(dim=2)[1]
skateboard_test_subset_predictions = skateboard_test_subset_predictions.detach().cpu().numpy()

table_test_subset_predictions = th.from_numpy(table_test_subset_predictions)
table_test_subset_predictions = table_test_subset_predictions.permute(0, 2, 1).contiguous()
table_test_subset_predictions = table_test_subset_predictions.max(dim=2)[1]
table_test_subset_predictions = table_test_subset_predictions.detach().cpu().numpy()


# part masks - match number of parts with color scheme
indices_airplane_part_0 = np.array(airplane_test_subset_predictions == 0)
indices_airplane_part_1 = np.array(airplane_test_subset_predictions == 1)
indices_airplane_part_2 = np.array(airplane_test_subset_predictions == 2)
indices_airplane_part_3 = np.array(airplane_test_subset_predictions == 3)

indices_bag_part_0 = np.array(bag_test_subset_predictions == 4)
indices_bag_part_1 = np.array(bag_test_subset_predictions == 5)

indices_cap_part_0 = np.array(cap_test_subset_predictions == 6)
indices_cap_part_1 = np.array(cap_test_subset_predictions == 7)

indices_car_part_0 = np.array(car_test_subset_predictions == 8)
indices_car_part_1 = np.array(car_test_subset_predictions == 9)
indices_car_part_2 = np.array(car_test_subset_predictions == 10)
indices_car_part_3 = np.array(car_test_subset_predictions == 11)

indices_chair_part_0 = np.array(chair_test_subset_predictions == 12)
indices_chair_part_1 = np.array(chair_test_subset_predictions == 13)
indices_chair_part_2 = np.array(chair_test_subset_predictions == 14)
# part skip
indices_earphone_part_0 = np.array(earphone_test_subset_predictions == 16)
indices_earphone_part_1 = np.array(earphone_test_subset_predictions == 17)
indices_earphone_part_2 = np.array(earphone_test_subset_predictions == 18)

indices_guitar_part_0 = np.array(guitar_test_subset_predictions == 19)
indices_guitar_part_1 = np.array(guitar_test_subset_predictions == 20)
indices_guitar_part_2 = np.array(guitar_test_subset_predictions == 21)

indices_knife_part_0 = np.array(knife_test_subset_predictions == 22)
indices_knife_part_1 = np.array(knife_test_subset_predictions == 23)

indices_lamp_part_0 = np.array(lamp_test_subset_predictions == 24)
indices_lamp_part_1 = np.array(lamp_test_subset_predictions == 25)
indices_lamp_part_2 = np.array(lamp_test_subset_predictions == 26)
indices_lamp_part_3 = np.array(lamp_test_subset_predictions == 27)

indices_laptop_part_0 = np.array(laptop_test_subset_predictions == 28)
indices_laptop_part_1 = np.array(laptop_test_subset_predictions == 29)

indices_motorbike_part_0 = np.array(motorbike_test_subset_predictions == 30)
indices_motorbike_part_1 = np.array(motorbike_test_subset_predictions == 31)
indices_motorbike_part_2 = np.array(motorbike_test_subset_predictions == 32)
indices_motorbike_part_3 = np.array(motorbike_test_subset_predictions == 33)
indices_motorbike_part_4 = np.array(motorbike_test_subset_predictions == 35)
# part skip
indices_mug_part_0 = np.array(mug_test_subset_predictions == 36)
indices_mug_part_1 = np.array(mug_test_subset_predictions == 37)

indices_pistol_part_0 = np.array(pistol_test_subset_predictions == 38)
indices_pistol_part_1 = np.array(pistol_test_subset_predictions == 39)
indices_pistol_part_2 = np.array(pistol_test_subset_predictions == 40)

indices_rocket_part_0 = np.array(rocket_test_subset_predictions == 41)
indices_rocket_part_1 = np.array(rocket_test_subset_predictions == 42)
indices_rocket_part_2 = np.array(rocket_test_subset_predictions == 43)

indices_skateboard_part_0 = np.array(skateboard_test_subset_predictions == 44)
indices_skateboard_part_1 = np.array(skateboard_test_subset_predictions == 45)
indices_skateboard_part_2 = np.array(skateboard_test_subset_predictions == 46)

indices_table_part_0 = np.array(table_test_subset_predictions == 47)
indices_table_part_1 = np.array(table_test_subset_predictions == 48)
indices_table_part_2 = np.array(table_test_subset_predictions == 49)

# applying part mask to get subsets for all 30 examples

airplane_conv_8_hidden_output_part_0 = []
airplane_conv_8_hidden_output_part_1 = []
airplane_conv_8_hidden_output_part_2 = []
airplane_conv_8_hidden_output_part_3 = []

bag_conv_8_hidden_output_part_0 = []
bag_conv_8_hidden_output_part_1 = []

cap_conv_8_hidden_output_part_0 = []
cap_conv_8_hidden_output_part_1 = []

car_conv_8_hidden_output_part_0 = []
car_conv_8_hidden_output_part_1 = []
car_conv_8_hidden_output_part_2 = []
car_conv_8_hidden_output_part_3 = []

chair_conv_8_hidden_output_part_0 = []
chair_conv_8_hidden_output_part_1 = []
chair_conv_8_hidden_output_part_2 = []

earphone_conv_8_hidden_output_part_0 = []
earphone_conv_8_hidden_output_part_1 = []
earphone_conv_8_hidden_output_part_2 = []

guitar_conv_8_hidden_output_part_0 = []
guitar_conv_8_hidden_output_part_1 = []
guitar_conv_8_hidden_output_part_2 = []

knife_conv_8_hidden_output_part_0 = []
knife_conv_8_hidden_output_part_1 = []

lamp_conv_8_hidden_output_part_0 = []
lamp_conv_8_hidden_output_part_1 = []
lamp_conv_8_hidden_output_part_2 = []
lamp_conv_8_hidden_output_part_3 = []

laptop_conv_8_hidden_output_part_0 = []
laptop_conv_8_hidden_output_part_1 = []

motorbike_conv_8_hidden_output_part_0 = []
motorbike_conv_8_hidden_output_part_1 = []
motorbike_conv_8_hidden_output_part_2 = []
motorbike_conv_8_hidden_output_part_3 = []
motorbike_conv_8_hidden_output_part_4 = []

mug_conv_8_hidden_output_part_0 = []
mug_conv_8_hidden_output_part_1 = []

pistol_conv_8_hidden_output_part_0 = []
pistol_conv_8_hidden_output_part_1 = []
pistol_conv_8_hidden_output_part_2 = []

rocket_conv_8_hidden_output_part_0 = []
rocket_conv_8_hidden_output_part_1 = []
rocket_conv_8_hidden_output_part_2 = []

skateboard_conv_8_hidden_output_part_0 = []
skateboard_conv_8_hidden_output_part_1 = []
skateboard_conv_8_hidden_output_part_2 = []

table_conv_8_hidden_output_part_0 = []
table_conv_8_hidden_output_part_1 = []
table_conv_8_hidden_output_part_2 = []

for i in range(airplane_test_subset_part_labels.shape[0]):
    
    # subsetting
    inner_result_airplane_0 = airplane_conv9_hidden_output[i][indices_airplane_part_0[i]]
    inner_result_airplane_1 = airplane_conv9_hidden_output[i][indices_airplane_part_1[i]]
    inner_result_airplane_2 = airplane_conv9_hidden_output[i][indices_airplane_part_2[i]]
    inner_result_airplane_3 = airplane_conv9_hidden_output[i][indices_airplane_part_3[i]]
    
    inner_result_bag_0 = bag_conv9_hidden_output[i][indices_bag_part_0[i]]
    inner_result_bag_1 = bag_conv9_hidden_output[i][indices_bag_part_1[i]]
    
    inner_result_cap_0 = cap_conv9_hidden_output[i][indices_cap_part_0[i]]
    inner_result_cap_1 = cap_conv9_hidden_output[i][indices_cap_part_1[i]]
    
    inner_result_car_0 = car_conv9_hidden_output[i][indices_car_part_0[i]]
    inner_result_car_1 = car_conv9_hidden_output[i][indices_car_part_1[i]]
    inner_result_car_2 = car_conv9_hidden_output[i][indices_car_part_2[i]]
    inner_result_car_3 = car_conv9_hidden_output[i][indices_car_part_3[i]]

    inner_result_chair_0 = chair_conv9_hidden_output[i][indices_chair_part_0[i]]
    inner_result_chair_1 = chair_conv9_hidden_output[i][indices_chair_part_1[i]]
    inner_result_chair_2 = chair_conv9_hidden_output[i][indices_chair_part_2[i]]
    
    inner_result_earphone_0 = earphone_conv9_hidden_output[i][indices_earphone_part_0[i]]
    inner_result_earphone_1 = earphone_conv9_hidden_output[i][indices_earphone_part_1[i]]
    inner_result_earphone_2 = earphone_conv9_hidden_output[i][indices_earphone_part_2[i]]
    
    inner_result_guitar_0 = guitar_conv9_hidden_output[i][indices_guitar_part_0[i]]
    inner_result_guitar_1 = guitar_conv9_hidden_output[i][indices_guitar_part_1[i]]
    inner_result_guitar_2 = guitar_conv9_hidden_output[i][indices_guitar_part_2[i]]
    
    inner_result_knife_0 = knife_conv9_hidden_output[i][indices_knife_part_0[i]]
    inner_result_knife_1 = knife_conv9_hidden_output[i][indices_knife_part_1[i]]
    
    inner_result_lamp_0 = lamp_conv9_hidden_output[i][indices_lamp_part_0[i]]
    inner_result_lamp_1 = lamp_conv9_hidden_output[i][indices_lamp_part_1[i]]
    inner_result_lamp_2 = lamp_conv9_hidden_output[i][indices_lamp_part_2[i]]
    inner_result_lamp_3 = lamp_conv9_hidden_output[i][indices_lamp_part_3[i]]
    
    inner_result_laptop_0 = laptop_conv9_hidden_output[i][indices_laptop_part_0[i]]
    inner_result_laptop_1 = laptop_conv9_hidden_output[i][indices_laptop_part_1[i]]
    
    inner_result_motorbike_0 = motorbike_conv9_hidden_output[i][indices_motorbike_part_0[i]]
    inner_result_motorbike_1 = motorbike_conv9_hidden_output[i][indices_motorbike_part_1[i]]
    inner_result_motorbike_2 = motorbike_conv9_hidden_output[i][indices_motorbike_part_2[i]]
    inner_result_motorbike_3 = motorbike_conv9_hidden_output[i][indices_motorbike_part_3[i]]
    inner_result_motorbike_4 = motorbike_conv9_hidden_output[i][indices_motorbike_part_4[i]]
    
    inner_result_mug_0 = mug_conv9_hidden_output[i][indices_mug_part_0[i]]
    inner_result_mug_1 = mug_conv9_hidden_output[i][indices_mug_part_1[i]]
    
    inner_result_pistol_0 = pistol_conv9_hidden_output[i][indices_pistol_part_0[i]]
    inner_result_pistol_1 = pistol_conv9_hidden_output[i][indices_pistol_part_1[i]]
    inner_result_pistol_2 = pistol_conv9_hidden_output[i][indices_pistol_part_2[i]]
    
    inner_result_rocket_0 = rocket_conv9_hidden_output[i][indices_rocket_part_0[i]]
    inner_result_rocket_1 = rocket_conv9_hidden_output[i][indices_rocket_part_1[i]]
    inner_result_rocket_2 = rocket_conv9_hidden_output[i][indices_rocket_part_2[i]]
    
    inner_result_skateboard_0 = skateboard_conv9_hidden_output[i][indices_skateboard_part_0[i]]
    inner_result_skateboard_1 = skateboard_conv9_hidden_output[i][indices_skateboard_part_1[i]]
    inner_result_skateboard_2 = skateboard_conv9_hidden_output[i][indices_skateboard_part_2[i]]
    
    inner_result_table_0 = table_conv9_hidden_output[i][indices_table_part_0[i]]
    inner_result_table_1 = table_conv9_hidden_output[i][indices_table_part_1[i]]
    inner_result_table_2 = table_conv9_hidden_output[i][indices_table_part_2[i]]
        
    # appending
    airplane_conv_8_hidden_output_part_0.append(inner_result_airplane_0)
    airplane_conv_8_hidden_output_part_1.append(inner_result_airplane_1)
    airplane_conv_8_hidden_output_part_2.append(inner_result_airplane_2)
    airplane_conv_8_hidden_output_part_3.append(inner_result_airplane_3)
    
    bag_conv_8_hidden_output_part_0.append(inner_result_bag_0)
    bag_conv_8_hidden_output_part_1.append(inner_result_bag_1)
    
    cap_conv_8_hidden_output_part_0.append(inner_result_cap_0)
    cap_conv_8_hidden_output_part_1.append(inner_result_cap_1)
    
    car_conv_8_hidden_output_part_0.append(inner_result_car_0)
    car_conv_8_hidden_output_part_1.append(inner_result_car_1)
    car_conv_8_hidden_output_part_2.append(inner_result_car_2)
    car_conv_8_hidden_output_part_3.append(inner_result_car_3)
    
    chair_conv_8_hidden_output_part_0.append(inner_result_chair_0)
    chair_conv_8_hidden_output_part_1.append(inner_result_chair_1)
    chair_conv_8_hidden_output_part_2.append(inner_result_chair_2)
    
    earphone_conv_8_hidden_output_part_0.append(inner_result_earphone_0)
    earphone_conv_8_hidden_output_part_1.append(inner_result_earphone_1)
    earphone_conv_8_hidden_output_part_2.append(inner_result_earphone_2)
    
    guitar_conv_8_hidden_output_part_0.append(inner_result_guitar_0)
    guitar_conv_8_hidden_output_part_1.append(inner_result_guitar_1)
    guitar_conv_8_hidden_output_part_2.append(inner_result_guitar_2)
 
    knife_conv_8_hidden_output_part_0.append(inner_result_knife_0)
    knife_conv_8_hidden_output_part_1.append(inner_result_knife_1)
    
    lamp_conv_8_hidden_output_part_0.append(inner_result_lamp_0)
    lamp_conv_8_hidden_output_part_1.append(inner_result_lamp_1)
    lamp_conv_8_hidden_output_part_2.append(inner_result_lamp_2)
    lamp_conv_8_hidden_output_part_3.append(inner_result_lamp_3)
    
    laptop_conv_8_hidden_output_part_0.append(inner_result_laptop_0)
    laptop_conv_8_hidden_output_part_1.append(inner_result_laptop_1)
    
    motorbike_conv_8_hidden_output_part_0.append(inner_result_motorbike_0)
    motorbike_conv_8_hidden_output_part_1.append(inner_result_motorbike_1)
    motorbike_conv_8_hidden_output_part_2.append(inner_result_motorbike_2)
    motorbike_conv_8_hidden_output_part_3.append(inner_result_motorbike_3)
    motorbike_conv_8_hidden_output_part_4.append(inner_result_motorbike_4)
    
    mug_conv_8_hidden_output_part_0.append(inner_result_mug_0)
    mug_conv_8_hidden_output_part_1.append(inner_result_mug_1)
    
    pistol_conv_8_hidden_output_part_0.append(inner_result_pistol_0)
    pistol_conv_8_hidden_output_part_1.append(inner_result_pistol_1)
    pistol_conv_8_hidden_output_part_2.append(inner_result_pistol_2)
    
    rocket_conv_8_hidden_output_part_0.append(inner_result_rocket_0)
    rocket_conv_8_hidden_output_part_1.append(inner_result_rocket_1)
    rocket_conv_8_hidden_output_part_2.append(inner_result_rocket_2)
    
    skateboard_conv_8_hidden_output_part_0.append(inner_result_skateboard_0)
    skateboard_conv_8_hidden_output_part_1.append(inner_result_skateboard_1)
    skateboard_conv_8_hidden_output_part_2.append(inner_result_skateboard_2)
    
    table_conv_8_hidden_output_part_0.append(inner_result_table_0)
    table_conv_8_hidden_output_part_1.append(inner_result_table_1)
    table_conv_8_hidden_output_part_2.append(inner_result_table_2)
    
# stacking lists
conv_hidden_output_baseline_subset = np.vstack((airplane_conv_8_hidden_output_part_0,
                                             airplane_conv_8_hidden_output_part_1,
                                             airplane_conv_8_hidden_output_part_2,
                                             airplane_conv_8_hidden_output_part_3,
                                             
                                             bag_conv_8_hidden_output_part_0,
                                             bag_conv_8_hidden_output_part_1,
                                             
                                             cap_conv_8_hidden_output_part_0,
                                             cap_conv_8_hidden_output_part_1,
                                             
                                             car_conv_8_hidden_output_part_0,
                                             car_conv_8_hidden_output_part_1,
                                             car_conv_8_hidden_output_part_2,
                                             car_conv_8_hidden_output_part_3,
                                            
                                             chair_conv_8_hidden_output_part_0,
                                             chair_conv_8_hidden_output_part_1,
                                             chair_conv_8_hidden_output_part_2,
                                            
                                             earphone_conv_8_hidden_output_part_0,
                                             earphone_conv_8_hidden_output_part_1,
                                             earphone_conv_8_hidden_output_part_2,
                                            
                                             guitar_conv_8_hidden_output_part_0,
                                             guitar_conv_8_hidden_output_part_1,
                                             guitar_conv_8_hidden_output_part_2,
                                            
                                             knife_conv_8_hidden_output_part_0,
                                             knife_conv_8_hidden_output_part_1,
                                            
                                             lamp_conv_8_hidden_output_part_0,
                                             lamp_conv_8_hidden_output_part_1,
                                             lamp_conv_8_hidden_output_part_2,
                                             lamp_conv_8_hidden_output_part_3,
                                            
                                             laptop_conv_8_hidden_output_part_0,
                                             laptop_conv_8_hidden_output_part_1,
                                            
                                             motorbike_conv_8_hidden_output_part_0,
                                             motorbike_conv_8_hidden_output_part_1,
                                             motorbike_conv_8_hidden_output_part_2,
                                             motorbike_conv_8_hidden_output_part_3,
                                             motorbike_conv_8_hidden_output_part_4,
                                            
                                             mug_conv_8_hidden_output_part_0,
                                             mug_conv_8_hidden_output_part_1,
                                            
                                             pistol_conv_8_hidden_output_part_0,
                                             pistol_conv_8_hidden_output_part_1,
                                             pistol_conv_8_hidden_output_part_2,
                                            
                                             rocket_conv_8_hidden_output_part_0,
                                             rocket_conv_8_hidden_output_part_1,
                                             rocket_conv_8_hidden_output_part_2,
                                            
                                             skateboard_conv_8_hidden_output_part_0,
                                             skateboard_conv_8_hidden_output_part_1,
                                             skateboard_conv_8_hidden_output_part_2,
                                            
                                             table_conv_8_hidden_output_part_0,
                                             table_conv_8_hidden_output_part_1,
                                             table_conv_8_hidden_output_part_2))

conv_hidden_output_baseline_subset = np.resize(conv_hidden_output_baseline_subset, (480,1)).flatten()

np.save('conv_hidden_output_baseline_subset.npy', conv_hidden_output_baseline_subset)

# def chamfer_distance(x, y, metric='l2', direction='bi'):
#     """Chamfer distance between two point clouds
#     Parameters
#     ----------
#     x: numpy array [n_points_x, n_dims]
#         first point cloud
#     y: numpy array [n_points_y, n_dims]
#         second point cloud
#     metric: string or callable, default ‘l2’
#         metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
#     direction: str
#         direction of Chamfer distance.
#             'y_to_x':  computes average minimal distance from every point in y to x
#             'x_to_y':  computes average minimal distance from every point in x to y
#             'bi': compute both
#     Returns
#     -------
#     chamfer_dist: float
#         computed bidirectional Chamfer distance:
#             sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
#     """
    
#     if direction=='y_to_x':
#         x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
#         min_y_to_x = x_nn.kneighbors(y)[0]
#         chamfer_dist = np.mean(min_y_to_x)
#     elif direction=='x_to_y':
#         y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
#         min_x_to_y = y_nn.kneighbors(x)[0]
#         chamfer_dist = np.mean(min_x_to_y)
#     elif direction=='bi':
#         x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
#         min_y_to_x = x_nn.kneighbors(y)[0]
#         y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
#         min_x_to_y = y_nn.kneighbors(x)[0]
#         chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
#     else:
#         raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
        
#     return chamfer_dist

# chamfer_dist_matrix_airplane_part_0 = np.asarray([[chamfer_distance(p1, p2) for p2 in airplane_conv_8_hidden_output_part_0] for p1 in airplane_conv_8_hidden_output_part_0])

# chamfer_distance(airplane_conv_8_hidden_output_part_0[0],airplane_conv_8_hidden_output_part_0[1])
# chamfer_distance(airplane_conv_8_hidden_output_part_0[3],airplane_conv_8_hidden_output_part_0[7])
# chamfer_distance(airplane_conv_8_hidden_output_part_0[5],airplane_conv_8_hidden_output_part_0[2])


##############################################################################


# lower the chamfer distance, more the similar the point clouds 
# good as absolute distance values vary between parent parts 

#getting the CD matrix for airplane parts from COllab saved output
cdm_baseline_subset = np.load('conv_9_chamfer_dist_matrix_baseline_subset_numpy.npy', allow_pickle = True)

#cdm_baseline_subset[0,5].item()

cdm_baseline_subset = np.vectorize(lambda x: x.item())(cdm_baseline_subset)
np.mean(cdm_baseline_subset)

# drop nan rows
cdm_baseline_subset = cdm_baseline_subset.flatten()
cdm_baseline_subset = cdm_baseline_subset[~np.isnan(cdm_baseline_subset)]
cdm_baseline_subset = np.reshape(cdm_baseline_subset, (435,435)) # take sqrt of stuff above

# COLOR MAPPING

# color map
# colormap = {"airplane_0": "gainsboro", "airplane_1": "darkgray", "airplane_2": "dimgray", "airplane_3": "black",
#             "car_0": "khaki", "car_1": "yellow", "car_2": "gold", "car_3": "goldenrod",
#             "guitar_0": "lightblue", "guitar_1": "deepskyblue", "guitar_2": "steelblue"
#             }
    
# colormap_inspect = predictions.map(colormap)
# colormap_inspect[colormap_inspect.isnull().any(0)]

# colormap manual - based on removal of NA values in CDM: 10x10x10x6 x 6x10x10x10 x 10x10x10
# since airplane_part_4 and car_part_0 have 4 empty examples each, giving NAs - manually inspect CDM to find out where these occur

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

# overall = 435 (matches)

# color map
# colormap = {"airplane_0": "gainsboro", "airplane_1": "darkgray", "airplane_2": "dimgray", "airplane_3": "black",
#             "bag_0": "lightcoral", "bag_1": "darkred",
#             "cap_0": "lightsalmon", "cap_1": "sienna",
#             "car_0": "khaki", "car_1": "yellow", "car_2": "gold", "car_3": "goldenrod",
#             "chair_0": "palegreen", "chair_1": "lawngreen", "chair_2": "yellowgreen", #, "olivedrab", # chair part
#             "earphone_0": "cyan", "earphone_1": "teal", "earphone_2": "darkslategray",
#             "guitar_0": "lightblue", "guitar_1": "deepskyblue", "guitar_2": "steelblue",
#             "knife_0": "wheat", "knife_1": "orange",
#             "lamp_0": "lightsteelblue", "lamp_1": "cornflowerblue", "lamp_2": "blue", "lamp_3": "navy",
#             "laptop_0": "lime", "laptop_1": "darkgreen",
#             "motor_0": "thistle", "motor_1": "plum", "motor_2": "violet", "motor_3": "mediumslateblue", "motor_4": "rebeccapurple", #  "mediumpurple" # secondlast color (motorbike part)
#             "mug_0": "hotpink", "mug_1": "deeppink",
#             # repeating here:
#             "pistol_0": "lightgray", "pistol_1": "dimgray", "pistol_2": "black",
#             "rocket_0": "khaki", "rocket_1": "yellow", "rocket_2": "gold",
#             "skateboard_0": "cyan", "skateboard_1": "teal", "skateboard_2": "darkslategray",
#             "table_0": "lightsteelblue", "table_1": "cornflowerblue", "table_2": "blue"
#             }

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

# applying UMAP to reduce the dimensionality to 2
embedding = reducer.fit_transform(cdm_baseline_subset)
embedding.shape       

# plotting
plt.figure(figsize=(8,6), dpi=500)
plt.scatter(embedding[:,0], embedding[:,1], s=7, marker = 'v', c=colormap)
plt.title('UMAP projection of the conv9 hidden output \n on the baseline subset - prediction separated', fontsize=24)

# without color
plt.figure(figsize=(8,6), dpi=180)
plt.scatter(embedding[:,0], embedding[:,1], s=5)
plt.title('UMAP projection of the conv9 hidden output \n on the baseline subset - prediction separated', fontsize=24)

###############################################################################

# reverse mapper to get part labels and labels again 

# saving embedding (random)
np.save('./Desktop/Master Thesis/Python experiments/THESIS FINAL RESULTS/Subsets/Part Seg/Conv9/conv9_chamfer_embedding_190_1.npy', embedding)

# reverse this colormap according to parts and labels
reverse_map_parts = {"#CFD8DC": 0, "#90A4AE": 1, "#607D8B": 2, "#455A64": 3,
            "#BDBDBD": 4, "#616161": 5,
            "#FF8A65": 6, "#E64A19": 7,
            "#BCAAA4": 8, "#8D6E63": 9, "#6D4C41": 10, "#4E342E": 11,
            "#FFE0B2": 12, "#FFB74D": 13, "#FB8C00": 14, #, "olivedrab", # chair part
            "#FFF9C4": 16, "#FFF176": 17, "#FDD835": 18,
            "#DCE775": 19, "#C0CA33": 20, "#9E9D24": 21,
            "#4CAF50": 22, "#1B5E20": 23,
            "#80CBC4": 24, "#26A69A": 25, "#00897B": 26, "#004D40": 27,
            "#80DEEA": 28, "#00BCD4": 29,
            "#BBDEFB": 30, "#90CAF9": 31, "#64B5F6": 32, "#2196F3": 33, "#1976D2": 35, "#0D47A1": 36,#  "mediumpurple" # secondlast color (motorbike part)
            "#3F51B5": 37, "#1A237E": 38,
            "#B39DDB": 39, "#7E57C2": 40, "#512DA8": 41,
            "#EF9A9A": 42, "#EF5350": 43, "#C62828": 44,
            "#F8BBD0": 45, "#F06292": 46, "#E91E63": 47,
            "#E1BEE7": 48, "#BA68C8": 49, "#9C27B0": 50
            }

# reverse this colormap according to parts and labels
reverse_map_labels = {"#CFD8DC": 0, "#90A4AE": 0, "#607D8B": 0, "#455A64": 0,
            "#BDBDBD": 1, "#616161": 1,
            "#FF8A65": 2, "#E64A19": 2,
            "#BCAAA4": 3, "#8D6E63": 3, "#6D4C41": 3, "#4E342E": 3,
            "#FFE0B2": 4, "#FFB74D": 4, "#FB8C00": 4, #, "olivedrab", # chair part
            "#FFF9C4": 5, "#FFF176": 5, "#FDD835": 5,
            "#DCE775": 6, "#C0CA33": 6, "#9E9D24": 6,
            "#4CAF50": 7, "#1B5E20": 7,
            "#80CBC4": 8, "#26A69A": 8, "#00897B": 8, "#004D40": 8,
            "#80DEEA": 9, "#00BCD4": 9,
            "#BBDEFB": 10, "#90CAF9": 10, "#64B5F6": 10, "#2196F3": 10, "#1976D2": 10, "#0D47A1": 10,#  "mediumpurple" # secondlast color (motorbike part)
            "#3F51B5": 11, "#1A237E": 11,
            "#B39DDB": 12, "#7E57C2": 12, "#512DA8": 12,
            "#EF9A9A": 13, "#EF5350": 13, "#C62828": 13,
            "#F8BBD0": 14, "#F06292": 14, "#E91E63": 14,
            "#E1BEE7": 15, "#BA68C8": 15, "#9C27B0": 15
            }


part_mapping = colormap.map(reverse_map_parts)
label_mapping = colormap.map(reverse_map_labels)
part_mapping.unique()
label_mapping.unique()

# saving part labels and labels 
np.save('./Desktop/Master Thesis/Python experiments/THESIS FINAL RESULTS/Subsets/Part Seg/Conv9/conv9_chamfer_predictions_parts.npy', part_mapping)
np.save('./Desktop/Master Thesis/Python experiments/THESIS FINAL RESULTS/Subsets/Part Seg/Conv9/conv9_chamfer_predictions_labels.npy', label_mapping)
