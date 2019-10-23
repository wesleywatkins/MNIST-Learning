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
        self.C = None
        self.min_data_point = None
        self.max_data_point = None

    def train(self, X, Y, trials):
        # set X and Y variables
        self.X = X
        self.Y = Y

        # get n amd m from data
        m = X.shape[0]
        n = X.shape[1]

        # get max and min (this will be useful later)
        self.min_data_point = np.min(X)
        self.max_data_point = np.max(X)

        # setup optimization variables
        w = cp.Variable((n, 1))
        b = cp.Variable()
        C = cp.Parameter(nonneg=True)

        # setup optimization problem
        loss = cp.sum(cp.pos(1 - cp.multiply(Y, X * w - b)))
        reg = 0.5 * cp.norm(w, 2)
        prob = cp.Problem(cp.Minimize(reg + C * loss))
        C_vals = np.logspace(-2, 0, trials)

        # solve
        w_vals = []
        b_vals = []
        train_error = np.zeros(trials)
        for i in range(trials):
            C.value = C_vals[i]
            prob.solve()
            train_error[i] = (self.Y != np.sign(X.dot(w.value) - b.value)).sum() / m
            w_vals.append(w.value)
            b_vals.append(b.value)

        # find smallest error
        min_error = train_error[0]
        self.w = w_vals[0]
        self.b = b_vals[0]
        self.C = C_vals[0]
        for (i, error) in enumerate(train_error):
            if error < min_error:
                min_error = error
                self.w = w_vals[i]
                self.b = b_vals[i]
                self.C = C_vals[i]

    def predict(self, features):
        if self.w is not None and self.b is not None:
            return np.sign(np.dot(np.array(features), self.w) + self.b)
        else:
            return 0

    # Got help from the following YouTube video
    # when it came to plotting the hyperplane
    # and support vectors
    # https://www.youtube.com/watch?v=yrnhziJk-z8&t=323s
    def visualize(self):
        # print data points
        for i in range(0, self.Y.size):
            if self.Y[i] == 1:
                plt.plot(self.X[i, 0], self.X[i, 1], 'bo')
            if self.Y[i] == -1:
                plt.plot(self.X[i, 0], self.X[i, 1], 'ro')

        # get decision boundary and support vectors
        dec = self.getDecBound()
        psv = self.getPosSupVec()
        nsv = self.getNegSupVec()

        # plot data
        plt.plot(dec[0], dec[1], 'k--')
        plt.plot(psv[0], psv[1], 'b')
        plt.plot(nsv[0], nsv[1], 'r')
        plt.show()

    def getLOOE(self):
        return len(self.getSupportVectors()) / self.Y.size

    def getSupportVectors(self):
        support_vectors = list()
        close_enough = 0.05  # this is to handle floating precision errors
        for i in range(self.Y.size):
            val = abs((np.dot(np.array(self.X[i]), self.w) + self.b))
            if 1 - close_enough <= val <= 1 + close_enough:
                support_vectors.append(self.X[i].tolist())
            if -1 - close_enough <= val <= -1 + close_enough:
                support_vectors.append(self.X[i].tolist())
        return support_vectors

    # Formula from: https://www.toppr.com/guides/maths/straight-lines/distance-of-point-from-a-line/
    def getMargin(self):
        c1 = self.hyperplane(0, self.w, self.b, 1)
        c2 = self.hyperplane(0, self.w, self.b, 0)
        points = self.getDecBound()
        slope = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0])
        return abs(c1 - c2) / (slope ** 2 + 1) ** 0.5

    def getDecBound(self):
        hyp_x_min = self.min_data_point * 0.9
        hyp_x_max = self.max_data_point * 1.1
        dec1 = self.hyperplane(hyp_x_min, self.w, self.b, 0)
        dec2 = self.hyperplane(hyp_x_max, self.w, self.b, 0)
        return [hyp_x_min, hyp_x_max], [dec1, dec2]

    def getPosSupVec(self):
        hyp_x_min = self.min_data_point * 0.9
        hyp_x_max = self.max_data_point * 1.1
        psv1 = self.hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = self.hyperplane(hyp_x_max, self.w, self.b, 1)
        return [hyp_x_min, hyp_x_max], [psv1, psv2]

    def getNegSupVec(self):
        hyp_x_min = self.min_data_point * 0.9
        hyp_x_max = self.max_data_point * 1.1
        nsv1 = self.hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = self.hyperplane(hyp_x_max, self.w, self.b, -1)
        return [hyp_x_min, hyp_x_max], [nsv1, nsv2]

    def hyperplane(self, x, w, b, v):
        return np.asscalar((-w[0] * x - b + v) / w[1])
