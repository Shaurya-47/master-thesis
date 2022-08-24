# Code for the master's thesis titled "Effective Activation Projections for Deep Learning on Point Clouds"

## Preliminaries for extraction of hidden layer activations:

1. In order to use the code from this repository to replicate the experiments of the master's thesis, it is first required to extract sample-wise activations (hidden layer representations) from a chosen neural network model designed for point cloud segmentation tasks.
2. In this master's thesis, the Dynamic Graph Convolutional Neural Network (DGCNN) model is selected for the purpose described in point 1: https://dl.acm.org/doi/10.1145/3326362. The following PyTorch implementation of the DGCNN model is selected: https://github.com/AnTao97/dgcnn.pytorch.
3. Replace the code in the `main_partseg.py` or `main_semseg.py` files in the repository listed in the point 2 with the code from `activation_extraction_partseg_conv9.py` or `activation_extraction_semseg_conv8.py` files from this repository respectively, depending on the part segmentation or semantic scene segmentation scenarios. The dataset for the part segmentation scenario is ***ShapeNet Part*** and the dataset for the semantic scene segmentation scenario is ***S3DIS***.
4. Thereafter, follow the instructions for the part segmentation or semantic scene segmentation scenarios provided in https://github.com/AnTao97/dgcnn.pytorch to run the evaluation scripts with model training or with a pre-trained model (after selecting a segmentation scenario). The respective code execution would lead to training of the DGCNN model (if the training mode is selected), evaluation on a test set, and storing of the corresponding hidden layer activations from the test set.

## Hyperparameters for the experiments:

1. Part segmentation
  * xyz 
