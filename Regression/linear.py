# Wesley Watkins
# wjw16

# Imports
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


# Class for linear regression model
class LinearRegression:

    # Initialize all variables to none
    def __init__(self):
        self.X = None
        self.Y = None
        self.w0 = None
        self.w1 = None

    # Training using cvxpy to find optimal w0 and w1
    def train(self, X, Y):
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
        self.X = X
        self.Y = Y
        self.w0 = w0.value
        self.w1 = w1.value

    # test model on given test data
    def test(self, X, Y):
        # skip if not trained
        if self.X is None or self.Y is None or self.w0 is None or self.w1 is None:
            return None
        # calculate # of misclassified points
        # in given test data
        misclassified = 0
        for i in range(0, Y.size):
            prediction = self.predict(X[i])
            actual = Y[i]
            if prediction != actual:
                misclassified += 1
        return misclassified

    def getLOOE(self):
        if self.X is None or self.Y is None:
            return None
        errors = []
        for i in range(self.Y.size):
            temp_X = np.copy(self.X)
            temp_Y = np.copy(self.Y)
            temp_X = np.delete(temp_X, i, 0)
            temp_Y = np.delete(temp_Y, i, 0)
            error = self.test(temp_X, temp_Y)
            errors.append(error/temp_Y.size)
        sum = 0
        for e in errors:
            sum += e
        return sum / len(errors)

    # predict + or - based off sign of w0 + w1*X
    def predict(self, features):
        if self.w0 is not None and self.w1 is not None:
            return np.sign(self.w0 + np.dot(features, self.w1))
        else:
            return 0

    # plot points and decision boundary
    def visualize(self):
        # skip if not trained
        if self.X is None or self.Y is None or self.w0 is None or self.w1 is None:
            return None

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
        x_min = np.min(self.X)
        x_max = np.max(self.X)
        dec1 = hyperplane(x_min, self.w1, self.w0, 0)
        dec2 = hyperplane(x_max, self.w1, self.w0, 0)
        plt.plot([x_min, x_max], [dec1, dec2], '--k')
        plt.show()
