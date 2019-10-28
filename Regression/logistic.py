import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self):
        self.X = None
        self.Y = None
        self.w = None
        self.min_value_feature = None
        self.max_value_feature = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, X, Y, l1_norm_weight=False):
        # variables
        self.X = X
        self.Y = Y
        m = X.shape[0]
        n = X.shape[1]
        # setup optimization problem
        w = cp.Variable((n, 1))
        cost = -(1/m) * np.sum(Y*np.log(self.sigmoid(np.dot(X, w))) + (1-Y) * np.log(1-self.sigmoid(np.dot(X, w))))
        reg = cp.norm(w, 1)
        if l1_norm_weight:
            prob = cp.Problem(cp.Minimize(reg + cost))
        else:
            prob = cp.Problem(cp.Minimize(cost))
        try:
            prob.solve()
        except:
            print("Uh oh")
        # set values
        self.w = w.value
        print("w: ", self.w)

    def predict(self, features):
        return self.sigmoid(np.dot(self.w, features))

    def visualize(self):
        for i in range(0, self.Y.size):
            if self.Y[i] == 1:
                plt.plot(self.X[i, 0], self.X[i, 1], 'bo')
            if self.Y[i] == -1:
                plt.plot(self.X[i, 0], self.X[i, 1], 'ro')