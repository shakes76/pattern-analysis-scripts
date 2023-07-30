'''
Test script for visualising the Mandelbrot set
'''
import numpy as np
import torch

# Version
print("TF Version:", torch.__version__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# grid for computing image, subdivide the space
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

# transfer to the GPU device
x = x.to(device)
y = y.to(device)

# Compute Gaussian
z = torch.exp(-(x**2+y**2)/2.0)

#plot
import matplotlib.pyplot as plt

plt.switch_backend('Qt5Agg')

plt.imshow(z.numpy())
plt.tight_layout()
plt.show()
