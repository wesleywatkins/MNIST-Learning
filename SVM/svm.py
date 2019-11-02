# Author: Wesley Watkins
# wjw16

# Import and Initializations
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


# create SVM class
class SupportVectorMachine:

    # initialize all variables to none
    def __init__(self):
        self.X = None
        self.Y = None
        self.w = None
        self.b = None
        self.C = None
        self.w_vals = None
        self.b_vals = None
        self.C_vals = None
        self.trials = None

    # train svm on given training data
    # also store the training data for plotting later
    def train(self, X, Y, trials, store_data=True):
        # set info from data
        if store_data:
            self.X, self.Y, self.trials = X, Y, trials
        m, n = X.shape[0], X.shape[1]
        # setup optimization variables
        w = cp.Variable((n, 1))
        b = cp.Variable()
        C = cp.Parameter(nonneg=True)
        # setup optimization problem
        loss = cp.sum(cp.pos(1 - cp.multiply(Y, X * w - b)))
        reg = 0.5 * cp.norm(w, 2)
        prob = cp.Problem(cp.Minimize(reg + C * loss))
        C_vals = np.logspace(-2, 0, trials)
        w_vals, b_vals = [], []
        min_error, min_error_i = 1, 0
        # try out a bunch of different C values to find the best one
        for i in range(trials):
            C.value = C_vals[i]
            prob.solve()
            train_error = (Y != np.sign(X.dot(w.value) - b.value)).sum() / m
            if train_error < min_error:
                min_error, min_error_i = train_error, i
            w_vals.append(w.value)
            b_vals.append(b.value)
        # find smallest error
        self.w = w_vals[min_error_i]
        self.b = b_vals[min_error_i]
        self.C = C_vals[min_error_i]
        # store all values
        self.w_vals = w_vals
        self.b_vals = b_vals
        self.C_vals = C_vals

    # predict + or - for a set of features
    def predict(self, features):
        if self.w is not None and self.b is not None:
            return np.sign(np.dot(np.array(features), self.w) + self.b)
        else:
            return 0

    # get test error on given test data
    def test(self, X, Y):
        misclassified = 0
        for i in range(0, Y.size):
            prediction, actual = self.predict(X[i]).item(), Y[i].item()
            if prediction != actual:
                misclassified += 1
        return misclassified

    # get test error on given test data
    # for a variety of different C values
    def testVariety(self, X, Y):
        best_w, best_b, best_C = self.w, self.b, self.C
        if len(self.C_vals) >= 5:
            random_indexes = np.random.choice(len(self.C_vals), 5, replace=False)
        else:
            random_indexes = range(0, len(self.C_vals))
        errors = np.zeros((len(random_indexes), 2))
        for (j, i) in enumerate(random_indexes):
            self.w = self.w_vals[i]
            self.b = self.b_vals[i]
            misclassified = 0
            for k in range(0, Y.size):
                prediction, actual = self.predict(X[k]), Y[k]
                if prediction != actual:
                    misclassified += 1
            errors[j][0] = self.C_vals[i]
            errors[j][1] = misclassified
        self.w, self.b, self.C = best_w, best_b, best_C
        return errors

    # Return a list of support vectors from the given training data
    def getSupportVectors(self):
        support_vectors = list()
        close_enough = 0.01  # this is to handle floating precision errors
        for i in range(self.Y.size):
            val = abs((np.dot(np.array(self.X[i]), self.w) + self.b))
            if 1 - close_enough <= val <= 1 + close_enough:
                support_vectors.append(self.X[i].tolist())
        return support_vectors

    # Formula from: https://www.toppr.com/guides/maths/straight-lines/distance-of-point-from-a-line/
    def getMargin(self):
        c1 = self.hyperplane(0, self.w, self.b, 1)
        c2 = self.hyperplane(0, self.w, self.b, 0)
        points = self.getDecBound()
        slope = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0])
        return abs(c1 - c2) / (slope ** 2 + 1) ** 0.5

    def getMarginVariety(self):
        best_w, best_b, best_C = self.w, self.b, self.C
        if len(self.C_vals) >= 5:
            random_indexes = np.random.choice(len(self.C_vals), 5, replace=False)
        else:
            random_indexes = range(0, len(self.C_vals))
        margins = np.zeros((len(random_indexes), 2))
        for (j, i) in enumerate(random_indexes):
            self.w = self.w_vals[i]
            self.b = self.b_vals[i]
            c1 = self.hyperplane(0, self.w, self.b, 1)
            c2 = self.hyperplane(0, self.w, self.b, 0)
            points = self.getDecBound()
            slope = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0])
            margins[j][0] = self.C_vals[i]
            margins[j][1] = abs(c1 - c2) / (slope ** 2 + 1) ** 0.5
        self.w, self.b, self.C = best_w, best_b, best_C
        return margins

    # LOOE <= SV / m+1
    def getLOOE(self, trials):
        # skip if not trained
        if self.X is None or self.Y is None:
            return None
        errors = []
        for i in range(self.Y.size):
            temp_X = np.copy(self.X)
            temp_Y = np.copy(self.Y)
            temp_X = np.delete(temp_X, i, 0)
            temp_Y = np.delete(temp_Y, i, 0)
            self.train(temp_X, temp_Y, trials, store_data=False)
            error = self.test(temp_X, temp_Y)
            errors.append(error / temp_Y.size)
        self.train(self.X, self.Y, self.trials)  # retrain on real data
        sum = 0
        for e in errors:
            sum += e
        return sum / len(errors)

    # Get two points from drawing decision boundary line
    def getDecBound(self):
        x_min = np.min(self.X) * 0.9
        x_max = np.max(self.X) * 1.1
        dec1 = self.hyperplane(x_min, self.w, self.b, 0)
        dec2 = self.hyperplane(x_max, self.w, self.b, 0)
        return [x_min, x_max], [dec1, dec2]

    # given a value of v, returns value from hyperplane
    # formula from https://www.youtube.com/watch?v=yrnhziJk-z8&t=323s
    def hyperplane(self, x, w, b, v):
        return np.asscalar((-w[0] * x - b + v) / w[1])

    # Got help from the following YouTube video
    # when it came to plotting the hyperplane
    # https://www.youtube.com/watch?v=yrnhziJk-z8&t=323s
    def visualize(self):
        # print data points
        for i in range(0, self.Y.size):
            if self.Y[i] == 1:
                plt.plot(self.X[i, 0], self.X[i, 1], 'bo')
            if self.Y[i] == -1:
                plt.plot(self.X[i, 0], self.X[i, 1], 'ro')
        # draw decision boundary
        dec = self.getDecBound()
        plt.plot(dec[0], dec[1], 'k', label="Decision Boundary")
        plt.legend()
        plt.show()

    # draw training points and decision boundary
    # for different values of C
    def visualizeVariety(self):
        # print data points
        for i in range(0, self.Y.size):
            if self.Y[i] == 1:
                plt.plot(self.X[i, 0], self.X[i, 1], 'bo')
            if self.Y[i] == -1:
                plt.plot(self.X[i, 0], self.X[i, 1], 'ro')
        # draw decision boundary for different values of C
        best_w = self.w
        best_b = self.b
        best_C = self.C
        if len(self.C_vals) >= 5:
            random_indexes = np.random.choice(len(self.C_vals), 5, replace=False)
        else:
            random_indexes = range(0, len(self.C_vals))
        for i in random_indexes:
            self.w = self.w_vals[i]
            self.b = self.b_vals[i]
            dec = self.getDecBound()
            plt.plot(dec[0], dec[1], c=np.random.rand(3, ), label=str(round(self.C_vals[i], 3)))
        self.w = best_w
        self.b = best_b
        self.C = best_C
        plt.legend()
        plt.show()
