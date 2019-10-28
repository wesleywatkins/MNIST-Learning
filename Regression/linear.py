import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self):
        self.X = None
        self.Y = None
        self.w0 = None
        self.w1 = None
        self.min_value_feature = None
        self.max_value_feature = None

    def train(self, X, Y):
        self.X = X
        self.Y = Y
        self.min_value_feature = np.min(X)
        self.max_value_feature = np.max(X)
        m = X.shape[0]
        n = X.shape[1]
        w0 = cp.Variable()
        w1 = cp.Variable((n, 1))
        loss = 1/m * cp.sum((Y - w0 - X * w1) ** 2)
        prob = cp.Problem(cp.Minimize(loss))
        prob.solve()
        print(w0.value, w1.value)
        self.w0 = w0.value
        self.w1 = w1.value

    def predict(self, features):
        return np.sign(self.w0 + np.dot(features, self.w1))

    def visualize(self):
        for i in range(0, self.Y.size):
            if self.Y[i] == 1:
                plt.plot(self.X[i, 0], self.X[i, 1], 'bo')
            if self.Y[i] == -1:
                plt.plot(self.X[i, 0], self.X[i, 1], 'ro')

        def hyperplane(x, w, b, v):
            return np.asscalar((-w[0] * x - b + v) / w[1])

        hyp_x_min = self.min_value_feature
        hyp_x_max = self.max_value_feature
        dec1 = hyperplane(hyp_x_min, self.w1, self.w0, 0)
        dec2 = hyperplane(hyp_x_max, self.w1, self.w0, 0)
        plt.plot([hyp_x_min, hyp_x_max], [dec1, dec2])
        plt.show()


if __name__ == '__main__':
    X = np.array([[-5, 1], [2, 2], [-3, 3], [4, 4], [-1, -1], [6, -2], [-3, -3], [-4, -4]])
    Y = np.array([[1], [1], [1], [1], [-1], [-1], [-1], [-1]])
    lr = LinearRegression()
    lr.train(X, Y)
    print(lr.predict(np.array([7, -7])))
    print(lr.predict(np.array([7, 7])))
    lr.visualize()
