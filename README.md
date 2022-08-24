# Code for the master's thesis titled "Effective Activation Projections for Deep Learning on Point Clouds"

## Requirements
* Python 3.7
* PyTorch 1.2
* CUDA 10.0
* Packages: glob, h5py, sklearn, plyfile, umap, pytorch3d, matplotlib, numpy, pandas, sklearn, and scipy

## Pre-requisite: extraction of hidden layer activations

1. In order to use the code from this repository to replicate the experiments of the master's thesis, it is first required to extract sample-wise activations (hidden layer representations) from a chosen neural network model designed for point cloud segmentation tasks.
2. In this master's thesis, the Dynamic Graph Convolutional Neural Network (DGCNN) model is selected for the purpose described in point 1: https://dl.acm.org/doi/10.1145/3326362. The following PyTorch implementation of the DGCNN model is selected: https://github.com/AnTao97/dgcnn.pytorch.
3. Replace the code in the `main_partseg.py` or `main_semseg.py` files in the repository listed in the point 2 with the code from `activation_extraction_partseg_conv9.py` or `activation_extraction_semseg_conv8.py` files from this repository respectively, depending on the part segmentation or semantic scene segmentation scenarios. The dataset for the part segmentation scenario is ***ShapeNet Part*** (test set: 160 examples x 1024 cloud size) and the dataset for the semantic scene segmentation scenario is ***S3DIS*** (test set: 100 examples x 2048 cloud size). 
4. Thereafter, follow the instructions for the part segmentation or semantic scene segmentation scenarios provided in https://github.com/AnTao97/dgcnn.pytorch to run the evaluation scripts with model training or with a pre-trained model (after selecting a segmentation scenario). The respective code execution would lead to training of the DGCNN model (if the training mode is selected), evaluation on a test set, and storing of the corresponding hidden layer activations from the test set.

## Additional notes

1. After obtaining all the hidden layer activations, accordingly edit the path at the beginning of each script to reflect the new files
2. The 9th convolutional layer and 8th convolutional layers are used by default for the part and semantic scene segmentation scenarios respectively, but they can be changed by replacing `conv9` in line 359 or `conv8` in line 358 of the `activation_extraction_partseg_conv9.py` or `activation_extraction_semseg_conv8.py` scripts with other layers respectively.
3. The scripts in this repository conduct all experinments and generate visualisations via prediction information from the DGCNN neural network model. Label information can also be used for the same purpose as it is also saved via the activation extraction scripts

