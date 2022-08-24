# Code for the master's thesis titled "Effective Activation Projections for Deep Learning on Point Clouds"

## Preliminaries for extraction of hidden layer activations:

1. In order to use the code from this repository to replicate the experiments of the master's thesis, it is first required to extract sample-wise activations (hidden layer representations) from a chosen neural network model designed for point cloud segmentation tasks.
2. In this master's thesis, the Dynamic Graph Convolutional Neural Network (DGCNN) model is selected for the purpose described in point 1: https://dl.acm.org/doi/10.1145/3326362. The following PyTorch implementation of the DGCNN model is selected: https://github.com/AnTao97/dgcnn.pytorch.
3. Replace the `main_partseg.py` or `main_semseg.py` files in the repository listed in the point 2 with the `activation_extraction_partseg_conv9.py` or `activation_extraction_semseg_conv8.py` from this repository respectively (depending on the part segmentation or semantic segmentation scenario). Thereafter, follow the instructions for the part segmentation scenario provided in https://github.com/AnTao97/dgcnn.pytorch in order to extract the hidden layer information under the part segmentation scenario (ShapeNet Part dataset).
