# Author: Wesley Watkins
# wjw16

# Import and Initializations
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


# create SVM class
class SupportVectorMachine:

    def __init__(self):
        # initialize all variables to none
        self.X = None
        self.Y = None
        self.w = None
        self.b = None

    def train(self, X, Y):
        # get n amd m from data
        m = X.shape[0]
        n = X.shape[1]
        # setup optimization variables
        w = cp.Variable((n, 1))
        b = cp.Variable()
        slack = cp.Variable((n, 1))

        # setup optimization problem
        self.w = cp.Variable((n, 1))
        self.b = cp.Variable()
        loss = cp.sum(cp.pos(1 - cp.multiply(self.Y, self.X * self.w - self.b)))
        reg = cp.norm(self.w, 2)
        lambd = cp.Parameter(nonneg=True)
        prob = cp.Problem(cp.Minimize(loss/self.Y.size + lambd*reg))
        prob.solve()

    def predict(self, features: list):
        if self.w is not None and self.b is not None:
            return np.sign(np.dot(np.array(features), self.w) + self.b)
        else:
            return 0

    def visualize(self):
        for i in range(0, self.Y.size):
            if self.Y[i] == 1:
                plt.plot(self.X[i, 0], self.X[i, 1], 'bo')
            if self.Y[i] == -1:
                plt.plot(self.X[i, 0], self.X[i, 1], 'ro')
        x = np.linspace(-5, 5, 100)
        y = self.w * self.X + self.b
        plt.plot(x, y, '-g')
