import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


class LogisticRegression:

    # Initialize all variables to zero
    def __init__(self):
        self.X = None
        self.Y = None
        self.w = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, X, Y, store_data=True):
        # skip this during LOOCVE checking
        temp_y = np.zeros(Y.shape[0])
        if store_data:
            self.X = X
            self.Y = Y
            # convert -1 and 1 y-values to 0 and 1 y-values for training
            for i in range(Y.size):
                temp_y[i] = np.round(self.sigmoid(Y[i, 0]))
        else:
            for i in range(Y.size):
                temp_y = np.zeros(Y.shape[0])
                for i in range(Y.size):
                    temp_y[i] = Y[i, 0]
        Y = temp_y
        # setup optimization problem
        n = X.shape[1]
        w = cp.Variable(n)
        log_likelihood = cp.sum(cp.multiply(Y, X @ w) - cp.logistic(1 + cp.exp(X @ w)))
        problem = cp.Problem(cp.Maximize(log_likelihood / n - cp.norm(w, 1)))
        problem.solve()
        # set values
        self.w = w.value

    # test model on given test data
    def test(self, X, Y):
        # skip if not trained
        if self.X is None or self.Y is None or self.w is None:
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

    # predict + or - for a set of features
    def predict(self, features):
        if self.w is not None:
            return np.sign(np.dot(self.w, features))
        else:
            return 0

    # get the leave one out cross validation error
    def getLOOE(self):
        # skip if not trained
        if self.X is None or self.Y is None:
            return None
        # save old values of w0 and w1
        best_w = self.w
        errors = []
        for i in range(self.Y.size):
            temp_X = np.copy(self.X)
            temp_Y = np.copy(self.Y)
            temp_X = np.delete(temp_X, i, 0)
            temp_Y = np.delete(temp_Y, i, 0)
            self.train(temp_X, temp_Y, store_data=False)
            error = self.test(temp_X, temp_Y)
            errors.append(error/temp_Y.size)
        self.w = best_w
        sum = 0
        for e in errors:
            sum += e
        return sum / len(errors)

    # plot the training sample points and the decision boundary
    def visualize(self):
        # skip if not trained
        if self.X is None or self.Y is None or self.w is None:
            return None

        # plot training sample points
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
        x_min = np.min(self.X) * 0.9
        x_max = np.max(self.X) * 1.1
        dec1 = hyperplane(x_min, self.w, 0, 0)
        dec2 = hyperplane(x_max, self.w, 0, 0)
        plt.plot([x_min, x_max], [dec1, dec2], 'k', label="Decision Boundary")
        plt.legend()
        plt.show()
