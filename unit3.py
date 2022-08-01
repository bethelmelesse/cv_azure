import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchinfo import summary            
from pytorchcv import load_mnist, plot_results           # We use pytorchcv helper to load all data we have talked about in the previous unit.

print(" ")

load_mnist()

# FULLY-CONNECTED DENSE NEURAL NEWORKS 

print(" ")