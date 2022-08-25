# importing libraries
import torch as th
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#%matplotlib inline

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

# defining the UMAP reducer
reducer = umap.UMAP(min_dist = 0.5, n_neighbors = 300, random_state = 1) # this is the original baseline
# applying UMAP to reduce the dimensionality to 2
embedding = reducer.fit_transform(conv9_hidden_output_subset)
print(embedding.shape)

# saving the conv9 embedding locally so I can load it directly next time
np.save('./Results/conv9_full_data_baseline_UMAP_embedding_hp_0.5_300_rs_1.npy', embedding)

# plotting baseline on UMAP

# processing labels
parts_airplane = np.resize(airplane_test_subset_part_labels, (10240,1))
parts_bag = np.resize(bag_test_subset_part_labels, (10240,1))
parts_cap = np.resize(cap_test_subset_part_labels, (10240,1))
parts_car = np.resize(car_test_subset_part_labels, (10240,1))
parts_chair = np.resize(chair_test_subset_part_labels, (10240,1))
parts_earphone = np.resize(earphone_test_subset_part_labels, (10240,1))
parts_guitar = np.resize(guitar_test_subset_part_labels, (10240,1))
parts_knife = np.resize(knife_test_subset_part_labels, (10240,1))
parts_lamp = np.resize(lamp_test_subset_part_labels, (10240,1))
parts_laptop = np.resize(laptop_test_subset_part_labels, (10240,1))
parts_motorbike = np.resize(motorbike_test_subset_part_labels, (10240,1))
parts_mug = np.resize(mug_test_subset_part_labels, (10240,1))
parts_pistol = np.resize(pistol_test_subset_part_labels, (10240,1))
parts_rocket = np.resize(rocket_test_subset_part_labels, (10240,1))
parts_skateboard = np.resize(skateboard_test_subset_part_labels, (10240,1))
parts_table = np.resize(table_test_subset_part_labels, (10240,1))

part_labels = np.vstack((parts_airplane,
                         parts_bag,
                         parts_cap,
                         parts_car,
                         parts_chair,
                         parts_earphone,
                         parts_guitar,
                         parts_knife,
                         parts_lamp,
                         parts_laptop,
                         parts_motorbike,
                         parts_mug,
                         parts_pistol,
                         parts_rocket,
                         parts_skateboard,
                         parts_table)).flatten()

#np.unique(part_labels)

# processing predictions
predictions = np.vstack((airplane_test_subset_predictions,  
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
                         table_test_subset_predictions))

predictions = th.from_numpy(predictions)
predictions = predictions.permute(0, 2, 1).contiguous()
predictions = predictions.max(dim=2)[1]
predictions = predictions.detach().cpu().numpy()
predictions = np.resize(predictions, (163840,1)).flatten()
#np.unique(predictions)
#np.unique(predictions).shape

# preprocessing to generate color map
predictions = predictions.tolist()
predictions = ["airplane_0" if x==0 else x for x in predictions]
predictions = ["airplane_1" if x==1 else x for x in predictions]
predictions = ["airplane_2" if x==2 else x for x in predictions]
predictions = ["airplane_3" if x==3 else x for x in predictions]

predictions = ["bag_0" if x==4 else x for x in predictions]
predictions = ["bag_1" if x==5 else x for x in predictions]

predictions = ["cap_0" if x==6 else x for x in predictions]
predictions = ["cap_1" if x==7 else x for x in predictions]

predictions = ["car_0" if x==8 else x for x in predictions]
predictions = ["car_1" if x==9 else x for x in predictions]
predictions = ["car_2" if x==10 else x for x in predictions]
predictions = ["car_3" if x==11 else x for x in predictions]

predictions = ["chair_0" if x==12 else x for x in predictions]
predictions = ["chair_1" if x==13 else x for x in predictions]
predictions = ["chair_2" if x==14 else x for x in predictions]
                         # part skip
predictions = ["earphone_0" if x==16 else x for x in predictions]
predictions = ["earphone_1" if x==17 else x for x in predictions]
predictions = ["earphone_2" if x==18 else x for x in predictions]

predictions = ["guitar_0" if x==19 else x for x in predictions]
predictions = ["guitar_1" if x==20 else x for x in predictions]
predictions = ["guitar_2" if x==21 else x for x in predictions]

predictions = ["knife_0" if x==22 else x for x in predictions]
predictions = ["knife_1" if x==23 else x for x in predictions]

predictions = ["lamp_0" if x==24 else x for x in predictions]
predictions = ["lamp_1" if x==25 else x for x in predictions]
predictions = ["lamp_2" if x==26 else x for x in predictions]
predictions = ["lamp_3" if x==27 else x for x in predictions]

predictions = ["laptop_0" if x==28 else x for x in predictions]
predictions = ["laptop_1" if x==29 else x for x in predictions]
                         
predictions = ["motor_0" if x==30 else x for x in predictions]
predictions = ["motor_1" if x==31 else x for x in predictions]
predictions = ["motor_2" if x==32 else x for x in predictions]
predictions = ["motor_3" if x==33 else x for x in predictions]
predictions = ["motor_4" if x==34 else x for x in predictions]
predictions = ["motor_5" if x==35 else x for x in predictions]
predictions = ["mug_0" if x==36 else x for x in predictions]
predictions = ["mug_1" if x==37 else x for x in predictions]

predictions = ["pistol_0" if x==38 else x for x in predictions]
predictions = ["pistol_1" if x==39 else x for x in predictions]
predictions = ["pistol_2" if x==40 else x for x in predictions]

predictions = ["rocket_0" if x==41 else x for x in predictions]
predictions = ["rocket_1" if x==42 else x for x in predictions]
predictions = ["rocket_2" if x==43 else x for x in predictions]

predictions = ["skateboard_0" if x==44 else x for x in predictions]
predictions = ["skateboard_1" if x==45 else x for x in predictions]
predictions = ["skateboard_2" if x==46 else x for x in predictions]

predictions = ["table_0" if x==47 else x for x in predictions]
predictions = ["table_1" if x==48 else x for x in predictions]
predictions = ["table_2" if x==49 else x for x in predictions]

predictions = np.array(predictions).flatten()
predictions = pd.Series(predictions)
#predictions.shape
#predictions.unique()
#predictions[predictions.isnull().any(0)] # NA detection

# same methodology can be applied to part labels if needed

# defining color map
colormap = {"airplane_0": "#CFD8DC", "airplane_1": "#90A4AE", "airplane_2": "#607D8B", "airplane_3": "#455A64",
            "bag_0": "#BDBDBD", "bag_1": "#616161",
            "cap_0": "#FF8A65", "cap_1": "#E64A19",
            "car_0": "#BCAAA4", "car_1": "#8D6E63", "car_2": "#6D4C41", "car_3": "#4E342E",
            "chair_0": "#FFE0B2", "chair_1": "#FFB74D", "chair_2": "#FB8C00", #, "olivedrab", # chair part
            "earphone_0": "#FFF9C4", "earphone_1": "#FFF176", "earphone_2": "#FDD835",
            "guitar_0": "#DCE775", "guitar_1": "#C0CA33", "guitar_2": "#9E9D24",
            "knife_0": "#4CAF50", "knife_1": "#1B5E20",
            "lamp_0": "#80CBC4", "lamp_1": "#26A69A", "lamp_2": "#00897B", "lamp_3": "#004D40",
            "laptop_0": "#80DEEA", "laptop_1": "#00BCD4",
            "motor_0": "#BBDEFB", "motor_1": "#90CAF9", "motor_2": "#64B5F6", "motor_3": "#2196F3", "motor_4": "#1976D2", "motor_5": "#0D47A1",#  "mediumpurple" # secondlast color (motorbike part)
            "mug_0": "#3F51B5", "mug_1": "#1A237E",
            "pistol_0": "#B39DDB", "pistol_1": "#7E57C2", "pistol_2": "#512DA8",
            "rocket_0": "#EF9A9A", "rocket_1": "#EF5350", "rocket_2": "#C62828",
            "skateboard_0": "#F8BBD0", "skateboard_1": "#F06292", "skateboard_2": "#E91E63",
            "table_0": "#E1BEE7", "table_1": "#BA68C8", "table_2": "#9C27B0"
            }

# obtaining activation projection visualization - baseline
plt.figure(figsize=(8,6), dpi=1000)
plt.scatter(embedding[:,0], embedding[:,1], s=0.05, c=predictions.map(colormap))
plt.title('UMAP baseline projection (colored via predictions) \n on 10 samples (size 1024) from all 16 categories', fontsize=24)
