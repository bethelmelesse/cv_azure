# Computer Vision (CV) - is a field that studies how computers can gain some degree of understanding from difital images and/or video.
# problems of CV - image classification, object detection, and segmentation
# MNIST dataset - grayscale images of handwreitten digits, 28 * 28 
# 3 * H * W
# mulitdimensional arrays are also called tensors
# The main difference between tensors in PyTorch and numpy arrays is that tensors support parallel operations on GPU, if it is available. Also, PyTorch offers additional functionality, such as automatic differentiation, when operating on tensors.

# IMPORT THE PACKAGES NEEDED
from matplotlib.transforms import Transform
import torch
import torchvision 
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor  

# The dataset object returns the data in the form of Python Imagine Library (PIL) images, 
# which we convert to tensors by passing a transform=ToTensor() parameter.
print(" ")

data_train = torchvision.datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

data_test = torchvision.datasets.MNIST(
    root='data',
    download=True,
    train=False,
    transform=ToTensor()
)

# VISUALIZING THE DATASET

fig,ax = plt.subplots(1,7)
for i in range(7):
    ax[i].imshow(data_train[i][0].view(28,28))
    ax[i].set_title(data_train[i][1])
    ax[i].axis('off')
plt.show()

# DATASET DATASTRUCTURE

print('Training samples: ', len(data_train))
print("Test samples: ", len(data_test))

print("Tensor size: ", data_train[0][0].size())
print("First 10 digits are: ", [data_train[i][1] for i in range (10)])

print("Min intensity value: ", data_train[0][0].min().item())
print("Max intensity value: ", data_train[0][0].max().item())

# Neural networks work with tensors, and before training any models we need to convert our dataset into a set of tensors. 

print(" ")