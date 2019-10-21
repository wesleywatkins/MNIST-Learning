# Author: Wesley Watkins
# wjw16

# Import and Initializations
import numpy as np
import matplotlib.pyplot as plt
from svm import SupportVectorMachine

# seed random number generator (last 8 of lib #)
np.random.seed(79019719)
n = 2
m = 200
u_pos = [-1, -1]
u_neg = [1, 1]
co_variance = [[1, 1], [1, 1]]

A = np.zeros((m, n))
B = np.zeros((m, n))
for i in range(0, m):
    val_pos = np.random.normal(u_pos, co_variance, (1, n))
    val_neg = np.random.normal(u_neg, co_variance, (1, n))
    A[i] = val_pos
    B[i] = val_neg

data = {1: A, -1: B}

svm = SupportVectorMachine()
svm.train(data, n)
svm.visualize()
