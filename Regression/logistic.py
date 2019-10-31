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

    def train(self, X, Y):
        # variables
        self.X = X
        self.Y = Y
        temp_y = np.zeros(Y.shape[0])
        for i in range(self.Y.size):
            temp_y[i] = np.round(self.sigmoid(Y[i, 0]))
        Y = temp_y
        temp_y = None  # clear up memory
        self.min_value_feature = np.min(X)
        self.max_value_feature = np.max(X)
        m = X.shape[0]
        n = X.shape[1]
        # useful functions
        w = cp.Variable(n)
        log_likelihood = cp.sum(
            cp.multiply(Y, X @ w) - cp.logistic(1 + cp.exp(X @ w))
        )
        problem = cp.Problem(cp.Maximize(log_likelihood / n - cp.norm(w, 1)))
        problem.solve()
        # set values
        self.w = w.value
        print(self.w)

    def predict(self, features):
        return np.sign(np.dot(self.w, features))

    def visualize(self):
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
        hyp_x_min = self.min_value_feature * 0.9
        hyp_x_max = self.max_value_feature * 1.1
        dec1 = hyperplane(hyp_x_min, self.w, 0, 0)
        dec2 = hyperplane(hyp_x_max, self.w, 0, 0)
        plt.plot([hyp_x_min, hyp_x_max], [dec1, dec2], '--k')
        plt.show()
