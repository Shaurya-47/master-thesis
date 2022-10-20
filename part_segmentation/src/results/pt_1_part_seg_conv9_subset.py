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

# getting object-wise hidden layer outputs and predictions
airplane_conv9_hidden_output_subset = conv9_hidden_output_subset[0:10]
bag_conv9_hidden_output_subset = conv9_hidden_output_subset[10:20]
cap_conv9_hidden_output_subset = conv9_hidden_output_subset[20:30]
car_conv9_hidden_output_subset = conv9_hidden_output_subset[30:40]
chair_conv9_hidden_output_subset = conv9_hidden_output_subset[40:50]
earphone_conv9_hidden_output_subset = conv9_hidden_output_subset[50:60]
guitar_conv9_hidden_output_subset = conv9_hidden_output_subset[60:70]
knife_conv9_hidden_output_subset = conv9_hidden_output_subset[70:80]
lamp_conv9_hidden_output_subset = conv9_hidden_output_subset[80:90]
laptop_conv9_hidden_output_subset = conv9_hidden_output_subset[90:100]
motorbike_conv9_hidden_output_subset = conv9_hidden_output_subset[100:110]
mug_conv9_hidden_output_subset = conv9_hidden_output_subset[110:120]
pistol_conv9_hidden_output_subset = conv9_hidden_output_subset[120:130]
rocket_conv9_hidden_output_subset = conv9_hidden_output_subset[130:140]
skateboard_conv9_hidden_output_subset = conv9_hidden_output_subset[140:150]
table_conv9_hidden_output_subset = conv9_hidden_output_subset[150:160]

airplane_preds_subset = preds_subset[0:10]
bag_preds_subset = preds_subset[10:20]
cap_preds_subset = preds_subset[20:30]
car_preds_subset = preds_subset[30:40]
chair_preds_subset = preds_subset[40:50]
earphone_preds_subset = preds_subset[50:60]
guitar_preds_subset = preds_subset[60:70]
knife_preds_subset = preds_subset[70:80]
lamp_preds_subset = preds_subset[80:90]
laptop_preds_subset = preds_subset[90:100]
motorbike_preds_subset = preds_subset[100:110]
mug_preds_subset = preds_subset[110:120]
pistol_preds_subset = preds_subset[120:130]
rocket_preds_subset = preds_subset[130:140]
skateboard_preds_subset = preds_subset[140:150]
table_preds_subset = preds_subset[150:160]

# restructuring object-wise hidden layer outputs and predictions
airplane_conv9_hidden_output_subset = np.resize(airplane_conv9_hidden_output_subset, (10,1024,256))
bag_conv9_hidden_output_subset = np.resize(bag_conv9_hidden_output_subset, (10,1024,256))
cap_conv9_hidden_output_subset = np.resize(cap_conv9_hidden_output_subset, (10,1024,256))
car_conv9_hidden_output_subset = np.resize(car_conv9_hidden_output_subset, (10,1024,256))
chair_conv9_hidden_output_subset = np.resize(chair_conv9_hidden_output_subset, (10,1024,256))
earphone_conv9_hidden_output_subset = np.resize(earphone_conv9_hidden_output_subset, (10,1024,256))
guitar_conv9_hidden_output_subset = np.resize(guitar_conv9_hidden_output_subset, (10,1024,256))
knife_conv9_hidden_output_subset = np.resize(knife_conv9_hidden_output_subset, (10,1024,256))
lamp_conv9_hidden_output_subset = np.resize(lamp_conv9_hidden_output_subset, (10,1024,256))
laptop_conv9_hidden_output_subset = np.resize(laptop_conv9_hidden_output_subset, (10,1024,256))
motorbike_conv9_hidden_output_subset = np.resize(motorbike_conv9_hidden_output_subset, (10,1024,256))
mug_conv9_hidden_output_subset = np.resize(mug_conv9_hidden_output_subset, (10,1024,256))
pistol_conv9_hidden_output_subset = np.resize(pistol_conv9_hidden_output_subset, (10,1024,256))
rocket_conv9_hidden_output_subset = np.resize(rocket_conv9_hidden_output_subset, (10,1024,256))
skateboard_conv9_hidden_output_subset = np.resize(skateboard_conv9_hidden_output_subset, (10,1024,256))
table_conv9_hidden_output_subset = np.resize(table_conv9_hidden_output_subset, (10,1024,256))

airplane_preds_subset = np.resize(airplane_preds_subset, (10,1024,50))
bag_preds_subset = np.resize(bag_preds_subset, (10,1024,50))
cap_preds_subset = np.resize(cap_preds_subset, (10,1024,50))
car_preds_subset = np.resize(car_preds_subset, (10,1024,50))
chair_preds_subset = np.resize(chair_preds_subset, (10,1024,50))
earphone_preds_subset = np.resize(earphone_preds_subset, (10,1024,50))
guitar_preds_subset = np.resize(guitar_preds_subset, (10,1024,50))
knife_preds_subset = np.resize(knife_preds_subset, (10,1024,50))
lamp_preds_subset = np.resize(lamp_preds_subset, (10,1024,50))
laptop_preds_subset = np.resize(laptop_preds_subset, (10,1024,50))
motorbike_preds_subset = np.resize(motorbike_preds_subset, (10,1024,50))
mug_preds_subset = np.resize(mug_preds_subset, (10,1024,50))
pistol_preds_subset = np.resize(pistol_preds_subset, (10,1024,50))
rocket_preds_subset = np.resize(rocket_preds_subset, (10,1024,50))
skateboard_preds_subset = np.resize(skateboard_preds_subset, (10,1024,50))
table_preds_subset = np.resize(table_preds_subset, (10,1024,50))

# dropping the one-hot encoding dimension for predictions
airplane_preds_subset = th.from_numpy(airplane_preds_subset).max(dim=2)[1].detach().cpu().numpy()
bag_preds_subset = th.from_numpy(bag_preds_subset).max(dim=2)[1].detach().cpu().numpy()
cap_preds_subset = th.from_numpy(cap_preds_subset).max(dim=2)[1].detach().cpu().numpy()
car_preds_subset = th.from_numpy(car_preds_subset).max(dim=2)[1].detach().cpu().numpy()
chair_preds_subset = th.from_numpy(chair_preds_subset).max(dim=2)[1].detach().cpu().numpy()
earphone_preds_subset = th.from_numpy(earphone_preds_subset).max(dim=2)[1].detach().cpu().numpy()
guitar_preds_subset = th.from_numpy(guitar_preds_subset).max(dim=2)[1].detach().cpu().numpy()
knife_preds_subset = th.from_numpy(knife_preds_subset).max(dim=2)[1].detach().cpu().numpy()
lamp_preds_subset = th.from_numpy(lamp_preds_subset).max(dim=2)[1].detach().cpu().numpy()
laptop_preds_subset = th.from_numpy(laptop_preds_subset).max(dim=2)[1].detach().cpu().numpy()
motorbike_preds_subset = th.from_numpy(motorbike_preds_subset).max(dim=2)[1].detach().cpu().numpy()
mug_preds_subset = th.from_numpy(mug_preds_subset).max(dim=2)[1].detach().cpu().numpy()
pistol_preds_subset = th.from_numpy(pistol_preds_subset).max(dim=2)[1].detach().cpu().numpy()
rocket_preds_subset = th.from_numpy(rocket_preds_subset).max(dim=2)[1].detach().cpu().numpy()
skateboard_preds_subset = th.from_numpy(skateboard_preds_subset).max(dim=2)[1].detach().cpu().numpy()
table_preds_subset = th.from_numpy(table_preds_subset).max(dim=2)[1].detach().cpu().numpy()

#################################################### PUT THIS STUFF IN UTILS

# creating boolean masks per part and per example
indices_airplane_part_0 = np.array(airplane_preds_subset == 0)
indices_airplane_part_1 = np.array(airplane_preds_subset == 1)
indices_airplane_part_2 = np.array(airplane_preds_subset == 2)
indices_airplane_part_3 = np.array(airplane_preds_subset == 3)
indices_bag_part_0 = np.array(bag_preds_subset == 4)
indices_bag_part_1 = np.array(bag_preds_subset == 5)
indices_cap_part_0 = np.array(cap_preds_subset == 6)
indices_cap_part_1 = np.array(cap_preds_subset == 7)
indices_car_part_0 = np.array(car_preds_subset == 8)
indices_car_part_1 = np.array(car_preds_subset == 9)
indices_car_part_2 = np.array(car_preds_subset == 10)
indices_car_part_3 = np.array(car_preds_subset == 11)
indices_chair_part_0 = np.array(chair_preds_subset == 12)
indices_chair_part_1 = np.array(chair_preds_subset == 13)
indices_chair_part_2 = np.array(chair_preds_subset == 14)
indices_earphone_part_0 = np.array(earphone_preds_subset == 16)
indices_earphone_part_1 = np.array(earphone_preds_subset == 17)
indices_earphone_part_2 = np.array(earphone_preds_subset == 18)
indices_guitar_part_0 = np.array(guitar_preds_subset == 19)
indices_guitar_part_1 = np.array(guitar_preds_subset == 20)
indices_guitar_part_2 = np.array(guitar_preds_subset == 21)
indices_knife_part_0 = np.array(knife_preds_subset == 22)
indices_knife_part_1 = np.array(knife_preds_subset == 23)
indices_lamp_part_0 = np.array(lamp_preds_subset == 24)
indices_lamp_part_1 = np.array(lamp_preds_subset == 25)
indices_lamp_part_2 = np.array(lamp_preds_subset == 26)
indices_lamp_part_3 = np.array(lamp_preds_subset == 27)
indices_laptop_part_0 = np.array(laptop_preds_subset == 28)
indices_laptop_part_1 = np.array(laptop_preds_subset == 29)
indices_motorbike_part_0 = np.array(motorbike_preds_subset == 30)
indices_motorbike_part_1 = np.array(motorbike_preds_subset == 31)
indices_motorbike_part_2 = np.array(motorbike_preds_subset == 32)
indices_motorbike_part_3 = np.array(motorbike_preds_subset == 33)
indices_motorbike_part_4 = np.array(motorbike_preds_subset == 35)
indices_mug_part_0 = np.array(mug_preds_subset == 36)
indices_mug_part_1 = np.array(mug_preds_subset == 37)
indices_pistol_part_0 = np.array(pistol_preds_subset == 38)
indices_pistol_part_1 = np.array(pistol_preds_subset == 39)
indices_pistol_part_2 = np.array(pistol_preds_subset == 40)
indices_rocket_part_0 = np.array(rocket_preds_subset == 41)
indices_rocket_part_1 = np.array(rocket_preds_subset == 42)
indices_rocket_part_2 = np.array(rocket_preds_subset == 43)
indices_skateboard_part_0 = np.array(skateboard_preds_subset == 44)
indices_skateboard_part_1 = np.array(skateboard_preds_subset == 45)
indices_skateboard_part_2 = np.array(skateboard_preds_subset == 46)
indices_table_part_0 = np.array(table_preds_subset == 47)
indices_table_part_1 = np.array(table_preds_subset == 48)
indices_table_part_2 = np.array(table_preds_subset == 49)

# applying part mask to get array subsets per example part

# placeholders
airplane_conv9_hidden_output_subset_part_0 = []
airplane_conv9_hidden_output_subset_part_1 = []
airplane_conv9_hidden_output_subset_part_2 = []
airplane_conv9_hidden_output_subset_part_3 = []
bag_conv9_hidden_output_subset_part_0 = []
bag_conv9_hidden_output_subset_part_1 = []
cap_conv9_hidden_output_subset_part_0 = []
cap_conv9_hidden_output_subset_part_1 = []
car_conv9_hidden_output_subset_part_0 = []
car_conv9_hidden_output_subset_part_1 = []
car_conv9_hidden_output_subset_part_2 = []
car_conv9_hidden_output_subset_part_3 = []
chair_conv9_hidden_output_subset_part_0 = []
chair_conv9_hidden_output_subset_part_1 = []
chair_conv9_hidden_output_subset_part_2 = []
earphone_conv9_hidden_output_subset_part_0 = []
earphone_conv9_hidden_output_subset_part_1 = []
earphone_conv9_hidden_output_subset_part_2 = []
guitar_conv9_hidden_output_subset_part_0 = []
guitar_conv9_hidden_output_subset_part_1 = []
guitar_conv9_hidden_output_subset_part_2 = []
knife_conv9_hidden_output_subset_part_0 = []
knife_conv9_hidden_output_subset_part_1 = []
lamp_conv9_hidden_output_subset_part_0 = []
lamp_conv9_hidden_output_subset_part_1 = []
lamp_conv9_hidden_output_subset_part_2 = []
lamp_conv9_hidden_output_subset_part_3 = []
laptop_conv9_hidden_output_subset_part_0 = []
laptop_conv9_hidden_output_subset_part_1 = []
motorbike_conv9_hidden_output_subset_part_0 = []
motorbike_conv9_hidden_output_subset_part_1 = []
motorbike_conv9_hidden_output_subset_part_2 = []
motorbike_conv9_hidden_output_subset_part_3 = []
motorbike_conv9_hidden_output_subset_part_4 = []
mug_conv9_hidden_output_subset_part_0 = []
mug_conv9_hidden_output_subset_part_1 = []
pistol_conv9_hidden_output_subset_part_0 = []
pistol_conv9_hidden_output_subset_part_1 = []
pistol_conv9_hidden_output_subset_part_2 = []
rocket_conv9_hidden_output_subset_part_0 = []
rocket_conv9_hidden_output_subset_part_1 = []
rocket_conv9_hidden_output_subset_part_2 = []
skateboard_conv9_hidden_output_subset_part_0 = []
skateboard_conv9_hidden_output_subset_part_1 = []
skateboard_conv9_hidden_output_subset_part_2 = []
table_conv9_hidden_output_subset_part_0 = []
table_conv9_hidden_output_subset_part_1 = []
table_conv9_hidden_output_subset_part_2 = []

for i in range(airplane_preds_subset.shape[0]):
    # subsetting
    inner_result_airplane_0 = airplane_conv9_hidden_output_subset[i][indices_airplane_part_0[i]]
    inner_result_airplane_1 = airplane_conv9_hidden_output_subset[i][indices_airplane_part_1[i]]
    inner_result_airplane_2 = airplane_conv9_hidden_output_subset[i][indices_airplane_part_2[i]]
    inner_result_airplane_3 = airplane_conv9_hidden_output_subset[i][indices_airplane_part_3[i]]
    inner_result_bag_0 = bag_conv9_hidden_output_subset[i][indices_bag_part_0[i]]
    inner_result_bag_1 = bag_conv9_hidden_output_subset[i][indices_bag_part_1[i]]
    inner_result_cap_0 = cap_conv9_hidden_output_subset[i][indices_cap_part_0[i]]
    inner_result_cap_1 = cap_conv9_hidden_output_subset[i][indices_cap_part_1[i]]
    inner_result_car_0 = car_conv9_hidden_output_subset[i][indices_car_part_0[i]]
    inner_result_car_1 = car_conv9_hidden_output_subset[i][indices_car_part_1[i]]
    inner_result_car_2 = car_conv9_hidden_output_subset[i][indices_car_part_2[i]]
    inner_result_car_3 = car_conv9_hidden_output_subset[i][indices_car_part_3[i]]
    inner_result_chair_0 = chair_conv9_hidden_output_subset[i][indices_chair_part_0[i]]
    inner_result_chair_1 = chair_conv9_hidden_output_subset[i][indices_chair_part_1[i]]
    inner_result_chair_2 = chair_conv9_hidden_output_subset[i][indices_chair_part_2[i]]
    inner_result_earphone_0 = earphone_conv9_hidden_output_subset[i][indices_earphone_part_0[i]]
    inner_result_earphone_1 = earphone_conv9_hidden_output_subset[i][indices_earphone_part_1[i]]
    inner_result_earphone_2 = earphone_conv9_hidden_output_subset[i][indices_earphone_part_2[i]]
    inner_result_guitar_0 = guitar_conv9_hidden_output_subset[i][indices_guitar_part_0[i]]
    inner_result_guitar_1 = guitar_conv9_hidden_output_subset[i][indices_guitar_part_1[i]]
    inner_result_guitar_2 = guitar_conv9_hidden_output_subset[i][indices_guitar_part_2[i]]
    inner_result_knife_0 = knife_conv9_hidden_output_subset[i][indices_knife_part_0[i]]
    inner_result_knife_1 = knife_conv9_hidden_output_subset[i][indices_knife_part_1[i]]
    inner_result_lamp_0 = lamp_conv9_hidden_output_subset[i][indices_lamp_part_0[i]]
    inner_result_lamp_1 = lamp_conv9_hidden_output_subset[i][indices_lamp_part_1[i]]
    inner_result_lamp_2 = lamp_conv9_hidden_output_subset[i][indices_lamp_part_2[i]]
    inner_result_lamp_3 = lamp_conv9_hidden_output_subset[i][indices_lamp_part_3[i]]
    inner_result_laptop_0 = laptop_conv9_hidden_output_subset[i][indices_laptop_part_0[i]]
    inner_result_laptop_1 = laptop_conv9_hidden_output_subset[i][indices_laptop_part_1[i]]
    inner_result_motorbike_0 = motorbike_conv9_hidden_output_subset[i][indices_motorbike_part_0[i]]
    inner_result_motorbike_1 = motorbike_conv9_hidden_output_subset[i][indices_motorbike_part_1[i]]
    inner_result_motorbike_2 = motorbike_conv9_hidden_output_subset[i][indices_motorbike_part_2[i]]
    inner_result_motorbike_3 = motorbike_conv9_hidden_output_subset[i][indices_motorbike_part_3[i]]
    inner_result_motorbike_4 = motorbike_conv9_hidden_output_subset[i][indices_motorbike_part_4[i]]
    inner_result_mug_0 = mug_conv9_hidden_output_subset[i][indices_mug_part_0[i]]
    inner_result_mug_1 = mug_conv9_hidden_output_subset[i][indices_mug_part_1[i]]
    inner_result_pistol_0 = pistol_conv9_hidden_output_subset[i][indices_pistol_part_0[i]]
    inner_result_pistol_1 = pistol_conv9_hidden_output_subset[i][indices_pistol_part_1[i]]
    inner_result_pistol_2 = pistol_conv9_hidden_output_subset[i][indices_pistol_part_2[i]]
    inner_result_rocket_0 = rocket_conv9_hidden_output_subset[i][indices_rocket_part_0[i]]
    inner_result_rocket_1 = rocket_conv9_hidden_output_subset[i][indices_rocket_part_1[i]]
    inner_result_rocket_2 = rocket_conv9_hidden_output_subset[i][indices_rocket_part_2[i]]
    inner_result_skateboard_0 = skateboard_conv9_hidden_output_subset[i][indices_skateboard_part_0[i]]
    inner_result_skateboard_1 = skateboard_conv9_hidden_output_subset[i][indices_skateboard_part_1[i]]
    inner_result_skateboard_2 = skateboard_conv9_hidden_output_subset[i][indices_skateboard_part_2[i]]
    inner_result_table_0 = table_conv9_hidden_output_subset[i][indices_table_part_0[i]]
    inner_result_table_1 = table_conv9_hidden_output_subset[i][indices_table_part_1[i]]
    inner_result_table_2 = table_conv9_hidden_output_subset[i][indices_table_part_2[i]]

    # appending
    airplane_conv9_hidden_output_subset_part_0.append(inner_result_airplane_0)
    airplane_conv9_hidden_output_subset_part_1.append(inner_result_airplane_1)
    airplane_conv9_hidden_output_subset_part_2.append(inner_result_airplane_2)
    airplane_conv9_hidden_output_subset_part_3.append(inner_result_airplane_3)
    bag_conv9_hidden_output_subset_part_0.append(inner_result_bag_0)
    bag_conv9_hidden_output_subset_part_1.append(inner_result_bag_1)
    cap_conv9_hidden_output_subset_part_0.append(inner_result_cap_0)
    cap_conv9_hidden_output_subset_part_1.append(inner_result_cap_1)
    car_conv9_hidden_output_subset_part_0.append(inner_result_car_0)
    car_conv9_hidden_output_subset_part_1.append(inner_result_car_1)
    car_conv9_hidden_output_subset_part_2.append(inner_result_car_2)
    car_conv9_hidden_output_subset_part_3.append(inner_result_car_3)
    chair_conv9_hidden_output_subset_part_0.append(inner_result_chair_0)
    chair_conv9_hidden_output_subset_part_1.append(inner_result_chair_1)
    chair_conv9_hidden_output_subset_part_2.append(inner_result_chair_2)
    earphone_conv9_hidden_output_subset_part_0.append(inner_result_earphone_0)
    earphone_conv9_hidden_output_subset_part_1.append(inner_result_earphone_1)
    earphone_conv9_hidden_output_subset_part_2.append(inner_result_earphone_2)
    guitar_conv9_hidden_output_subset_part_0.append(inner_result_guitar_0)
    guitar_conv9_hidden_output_subset_part_1.append(inner_result_guitar_1)
    guitar_conv9_hidden_output_subset_part_2.append(inner_result_guitar_2)
    knife_conv9_hidden_output_subset_part_0.append(inner_result_knife_0)
    knife_conv9_hidden_output_subset_part_1.append(inner_result_knife_1)
    lamp_conv9_hidden_output_subset_part_0.append(inner_result_lamp_0)
    lamp_conv9_hidden_output_subset_part_1.append(inner_result_lamp_1)
    lamp_conv9_hidden_output_subset_part_2.append(inner_result_lamp_2)
    lamp_conv9_hidden_output_subset_part_3.append(inner_result_lamp_3)
    laptop_conv9_hidden_output_subset_part_0.append(inner_result_laptop_0)
    laptop_conv9_hidden_output_subset_part_1.append(inner_result_laptop_1)
    motorbike_conv9_hidden_output_subset_part_0.append(inner_result_motorbike_0)
    motorbike_conv9_hidden_output_subset_part_1.append(inner_result_motorbike_1)
    motorbike_conv9_hidden_output_subset_part_2.append(inner_result_motorbike_2)
    motorbike_conv9_hidden_output_subset_part_3.append(inner_result_motorbike_3)
    motorbike_conv9_hidden_output_subset_part_4.append(inner_result_motorbike_4)
    mug_conv9_hidden_output_subset_part_0.append(inner_result_mug_0)
    mug_conv9_hidden_output_subset_part_1.append(inner_result_mug_1)
    pistol_conv9_hidden_output_subset_part_0.append(inner_result_pistol_0)
    pistol_conv9_hidden_output_subset_part_1.append(inner_result_pistol_1)
    pistol_conv9_hidden_output_subset_part_2.append(inner_result_pistol_2)
    rocket_conv9_hidden_output_subset_part_0.append(inner_result_rocket_0)
    rocket_conv9_hidden_output_subset_part_1.append(inner_result_rocket_1)
    rocket_conv9_hidden_output_subset_part_2.append(inner_result_rocket_2)
    skateboard_conv9_hidden_output_subset_part_0.append(inner_result_skateboard_0)
    skateboard_conv9_hidden_output_subset_part_1.append(inner_result_skateboard_1)
    skateboard_conv9_hidden_output_subset_part_2.append(inner_result_skateboard_2)
    table_conv9_hidden_output_subset_part_0.append(inner_result_table_0)
    table_conv9_hidden_output_subset_part_1.append(inner_result_table_1)
    table_conv9_hidden_output_subset_part_2.append(inner_result_table_2)
    
# stacking lists
chamfer_distance_input = np.vstack((airplane_conv9_hidden_output_subset_part_0,
                                    airplane_conv9_hidden_output_subset_part_1,
                                    airplane_conv9_hidden_output_subset_part_2,
                                    airplane_conv9_hidden_output_subset_part_3,
                                    bag_conv9_hidden_output_subset_part_0,
                                    bag_conv9_hidden_output_subset_part_1,
                                    cap_conv9_hidden_output_subset_part_0,
                                    cap_conv9_hidden_output_subset_part_1,
                                    car_conv9_hidden_output_subset_part_0,
                                    car_conv9_hidden_output_subset_part_1,
                                    car_conv9_hidden_output_subset_part_2,
                                    car_conv9_hidden_output_subset_part_3,
                                    chair_conv9_hidden_output_subset_part_0,
                                    chair_conv9_hidden_output_subset_part_1,
                                    chair_conv9_hidden_output_subset_part_2,
                                    earphone_conv9_hidden_output_subset_part_0,
                                    earphone_conv9_hidden_output_subset_part_1,
                                    earphone_conv9_hidden_output_subset_part_2,
                                    guitar_conv9_hidden_output_subset_part_0,
                                    guitar_conv9_hidden_output_subset_part_1,
                                    guitar_conv9_hidden_output_subset_part_2,
                                    knife_conv9_hidden_output_subset_part_0,
                                    knife_conv9_hidden_output_subset_part_1,
                                    lamp_conv9_hidden_output_subset_part_0,
                                    lamp_conv9_hidden_output_subset_part_1,
                                    lamp_conv9_hidden_output_subset_part_2,
                                    lamp_conv9_hidden_output_subset_part_3,
                                    laptop_conv9_hidden_output_subset_part_0,
                                    laptop_conv9_hidden_output_subset_part_1,
                                    motorbike_conv9_hidden_output_subset_part_0,
                                    motorbike_conv9_hidden_output_subset_part_1,
                                    motorbike_conv9_hidden_output_subset_part_2,
                                    motorbike_conv9_hidden_output_subset_part_3,
                                    motorbike_conv9_hidden_output_subset_part_4,
                                    mug_conv9_hidden_output_subset_part_0,
                                    mug_conv9_hidden_output_subset_part_1,
                                    pistol_conv9_hidden_output_subset_part_0,
                                    pistol_conv9_hidden_output_subset_part_1,
                                    pistol_conv9_hidden_output_subset_part_2,
                                    rocket_conv9_hidden_output_subset_part_0,
                                    rocket_conv9_hidden_output_subset_part_1,
                                    rocket_conv9_hidden_output_subset_part_2,
                                    skateboard_conv9_hidden_output_subset_part_0,
                                    skateboard_conv9_hidden_output_subset_part_1,
                                    skateboard_conv9_hidden_output_subset_part_2,
                                    table_conv9_hidden_output_subset_part_0,
                                    table_conv9_hidden_output_subset_part_1,
                                    table_conv9_hidden_output_subset_part_2
                                    ))

chamfer_distance_input = np.resize(chamfer_distance_input, (480,1)).flatten()


# Chamfer distance: lower the chamfer distance, more the similar the point clouds 

# chamfer distance matrix calculation using Pytorch3D
chamfer_dist_matrix_baseline_subset_numpy = np.asarray([[list(cd(th.from_numpy(np.resize(p1, (1,p1.shape[0],256))).cuda(), th.from_numpy(np.resize(p2, (1,p2.shape[0],256))).cuda()))[0].cpu() for p2 in chamfer_distance_input] for p1 in chamfer_distance_input])

#getting the Chamfer Distance matrix between all the test activations
cdm_baseline_subset = np.vectorize(lambda x: x.item())(chamfer_dist_matrix_baseline_subset_numpy)
#np.mean(cdm_baseline_subset)
# drop nan rows
cdm_baseline_subset = cdm_baseline_subset.flatten()
cdm_baseline_subset = cdm_baseline_subset[~np.isnan(cdm_baseline_subset)]
cdm_baseline_subset = np.reshape(cdm_baseline_subset, (435,435)) # take sqrt of number above

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
