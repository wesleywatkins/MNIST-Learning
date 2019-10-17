# Author: Wesley Watkins
# wjw16

# Import and Initializations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from svm import SVM
style.use('ggplot')

# setup data dictionary
data = dict()

# Count Data
data_count = 0
with open('toyData.txt') as f:
    for line in f:
        if line[0] == '#':
            continue
        line = line.rstrip()
        temp = line.split(' ')
        if len(temp) != 3:
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