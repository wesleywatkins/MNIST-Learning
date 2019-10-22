# Author: Wesley Watkins
# wjw16

# Import and Initializations
import numpy as np
import matplotlib.pyplot as plt
import data as d
from svm import SupportVectorMachine

# Generate sample data
print("Generating data...")
X, Y = d.gen_data()
print("Data generated!")

for i in range(Y.size):
    if Y[i, 0] == 1:
        plt.plot(X[i, 0], X[i, 1], '-bo')
    else:
        plt.plot(X[i, 0], X[i, 1], '-ro')

plt.show()

# Create support vector machine
'''
svm = SupportVectorMachine()
svm.train(data, n)
svm.visualize()
'''