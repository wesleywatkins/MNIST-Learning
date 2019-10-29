# Wesley Watkins
# wjw16

# Imports
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


# Class for linear regression model
class LinearRegression:

    # Initialize all variables to zero
    def __init__(self):
        self.X = None
        self.Y = None
        self.w0 = None
        self.w1 = None
        # useful for plotting
        self.min_value_feature = None
        self.max_value_feature = None

    # Training using cvxpy to find optimal w0 and w1
    def train(self, X, Y):
        # store data
        self.X = X
        self.Y = Y
        # store min and max for plotting later
        self.min_value_feature = np.min(X)
        self.max_value_feature = np.max(X)
        # get sample and feature size
        m = X.shape[0]
        n = X.shape[1]
        # setup optimization problem
        w0 = cp.Variable()
        w1 = cp.Variable((n, 1))
        loss = 1/m * cp.sum((Y - w0 - X * w1) ** 2)
        # solve problem
        prob = cp.Problem(cp.Minimize(loss))
        prob.solve()
        # store values
        self.w0 = w0.value
        self.w1 = w1.value

    # predict + or - based off sign of w0 + w1*X
    def predict(self, features):
        return np.sign(self.w0 + np.dot(features, self.w1))

    # plot points and decision boundary
    def visualize(self):
        # plot training points
        for i in range(0, self.Y.size):
            if self.Y[i] == 1:
                plt.plot(self.X[i, 0], self.X[i, 1], 'bo')
            if self.Y[i] == -1:
                plt.plot(self.X[i, 0], self.X[i, 1], 'ro')

        # embedded function for computing decisionary boundary
        # Formula from https://www.youtube.com/watch?v=yrnhziJk-z8&t=323s
        def hyperplane(x, w, b, v):
            return np.asscalar((-w[0] * x - b + v) / w[1])

        # Get two points for drawing decision boundary
        hyp_x_min = self.min_value_feature
        hyp_x_max = self.max_value_feature
        dec1 = hyperplane(hyp_x_min, self.w1, self.w0, 0)
        dec2 = hyperplane(hyp_x_max, self.w1, self.w0, 0)
        plt.plot([hyp_x_min, hyp_x_max], [dec1, dec2], '--k')
        plt.show()
