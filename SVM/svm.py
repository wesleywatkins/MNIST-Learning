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

    def train(self, X, Y, trials):
        # set X and Y variables
        self.X = X
        self.Y = Y
        # get n amd m from data
        m = X.shape[0]
        n = X.shape[1]
        # setup optimization variables
        w = cp.Variable((n, 1))
        b = cp.Variable()
        slack = cp.Variable((n, 1))
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
        for (i, error) in enumerate(train_error):
            if error < min_error:
                min_error = error
                self.w = w_vals[i]
                self.b = b_vals[i]

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

        for i in range(0, self.Y.size):
            if self.Y[i] == 1:
                plt.plot(self.X[i, 0], self.X[i, 1], 'bo')
            if self.Y[i] == -1:
                plt.plot(self.X[i, 0], self.X[i, 1], 'ro')

        def hyperplane(x, w, b, v):
            return np.asscalar((-w[0] * x - b + v) / w[1])

        hyp_x_min = -4
        hyp_x_max = 4
        sig_figs = 4

        # compute decision boundary points
        dec1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        dec2 = hyperplane(hyp_x_max, self.w, self.b, 0)

        # compute positive support vector points
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)

        # compute negative support vector points
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)

        # compute positive support vector line
        slope = (psv2 - hyp_x_max) / (psv1 - hyp_x_min)
        y_intercept = hyperplane(0, self.w, self.b, 1)
        print("Positive Support Vector: y = ", round(slope, sig_figs), "x + ", round(y_intercept, sig_figs))

        # compute positive support vector line
        slope = (nsv2 - hyp_x_max) / (nsv1 - hyp_x_min)
        y_intercept = hyperplane(0, self.w, self.b, -1)
        print("Negative Support Vector: y = ", round(slope, sig_figs), "x + ", round(y_intercept, sig_figs))

        # compute margin
        # Formula from: https://www.toppr.com/guides/maths/straight-lines/distance-of-point-from-a-line/
        c1 = hyperplane(0, self.w, self.b, 1)
        c2 = hyperplane(0, self.w, self.b, 0)
        slope = (dec2 - hyp_x_max) / (dec1 - hyp_x_min)
        margin = abs(c1 - c2) / (slope**2 + 1) ** 0.5
        print("Margin: ", round(margin, sig_figs))

        # Plots
        plt.plot([hyp_x_min, hyp_x_max], [dec1, dec2], 'k--')
        plt.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'b')
        plt.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'r')

        plt.show()
