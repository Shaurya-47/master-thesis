# library imports
import torch as th
import numpy as np
import umap
import matplotlib.pyplot as plt
import pandas as pd
from pytorch3d.loss import chamfer_distance as cd

# importing data
conv9_hidden_output_subset = th.load('./data/conv9_hidden_output_subset.pt')
preds_subset = th.load('./data/preds_subset.pt')
part_labels_subset = th.load('./data/part_labels_subset.pt')

### reconvert to original form here - if it is really needed; OTHER WAY?????


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

conv9_hidden_output_baseline_subset = np.resize(conv_hidden_output_baseline_subset, (480,1)).flatten()

np.save('conv9_hidden_output_baseline_subset.npy', conv9_hidden_output_baseline_subset)

# Chamfer distance: lower the chamfer distance, more the similar the point clouds 

# chamfer distance matrix calculation using Pytorch3D
chamfer_dist_matrix_baseline_subset_numpy = np.asarray([[list(cd(th.from_numpy(np.resize(p1, (1,p1.shape[0],256))).cuda(), th.from_numpy(np.resize(p2, (1,p2.shape[0],256))).cuda()))[0].cpu() for p2 in conv9_hidden_output_baseline_subset] for p1 in conv9_hidden_output_baseline_subset])

#chamfer_dist_matrix_baseline_subset_numpy = np.asarray(chamfer_dist_matrix_baseline_subset)
np.save('./Results/conv9_chamfer_dist_matrix_full_data_predictions_numpy_cuda.npy', chamfer_dist_matrix_baseline_subset_numpy)
#print(chamfer_dist_matrix_baseline_subset_numpy.shape)

#getting the Chamfer Distance matrix between all the test activations
cdm_baseline_subset = np.vectorize(lambda x: x.item())(chamfer_dist_matrix_baseline_subset_numpy)
#np.mean(cdm_baseline_subset)
# drop nan rows
cdm_baseline_subset = cdm_baseline_subset.flatten()
cdm_baseline_subset = cdm_baseline_subset[~np.isnan(cdm_baseline_subset)]
cdm_baseline_subset = np.reshape(cdm_baseline_subset, (435,435)) # take sqrt of number above

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
