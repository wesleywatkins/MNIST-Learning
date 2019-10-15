# Author: Wesley Watkins
# wjw16

# Import and Initializations
import cvxpy
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# Count Data
data_count = 0
with open('toyData.txt') as f:
    for line in f:
        if line[0] == '#':
            continue
        data_count += 1

# Array Setup
x_values = np.zeros((data_count, 2))
y_values = np.zeros(data_count)

# Parse and Read Data
index = 0
with open('toyData.txt') as f:
    for line in f:
        if line[0] == '#':
            continue
        line = line.rstrip()
        temp = line.split(' ')
        if len(temp) == 3:
            x_values[index, 0] = int(temp[0])
            x_values[index, 1] = int(temp[1])
            y_values[index] = int(temp[2])
            index += 1

# plot the data
for i in range(0, data_count):
    if y_values[i] == -1:
        plt.plot(x_values[i, 0], x_values[i, 1], 'ro')
    else:
        plt.plot(x_values[i, 0], x_values[i, 1], 'bo')
plt.show()


# create SVM class
class SVM:
    def __init__(self):
        self.data = None
        self.w = None
        self.min_value = None
        self.max_value = None
        self.features = None
        self.labels = None

    def train(self, features, labels):
        self.features = features
        self.labels = labels
        opt_dict = {}
        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
        # get min and max feature values
        self.max_value = np.min(features)
        self.min_value = np.max(features)
        step_sizes = [self.max_value * 0.1, self.max_value * 0.01, self.max_value * 0.001]
        b_range_multiple = 5


    def predict(self, features):
        return np.sign(np.dot(np.array(features), self.w) + self.b)
