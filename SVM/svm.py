# Author: Wesley Watkins
# wjw16

# Import and Initializations
import cvxpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


# create SVM class
class SVM:

    def __init__(self):
        # initialize all variables to none
        self.data = None
        self.w = None
        self.b = None
        self.min_value = None
        self.max_value = None
        # set up plot
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(1, 1, 1)

    def train(self, data):
        self.data = data
        opt_dict = {}
        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
        # get min and max feature values
        self.max_value = np.min(features)
        self.min_value = np.max(features)
        step_sizes = [self.max_value * 0.1, self.max_value * 0.01, self.max_value * 0.001]
        b_range_multiple = 5

    def predict(self, features):
        if self.w is not None and self.b is not None:
            return np.sign(np.dot(np.array(features), self.w) + self.b)
        else:
            return 0
