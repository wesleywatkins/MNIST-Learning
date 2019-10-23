# Wesley Watkins
# wjw16

# imports
from svm import SupportVectorMachine
from mlxtend.data import loadlocal_mnist
import os
import numpy as np

# read in MNIST dataset
print("Reading in MNIST dataset...")
dir_path = os.path.dirname(os.path.realpath(__file__))
X, Y = loadlocal_mnist(
    images_path=os.path.join(dir_path, '../Samples/t10k-images.idx3-ubyte'),
    labels_path=os.path.join(dir_path, '../Samples/t10k-labels.idx1-ubyte')
)
print("Reading complete!")

# keep only 0s and 1s
print("\nRemoving non 0-1 labels from dataset...")

print("Non 0-1 labels removed from dataset!")
print(X.shape, Y.shape)
print(Y)