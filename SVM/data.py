import numpy as np


def gen_data(seed=79019719):
    # variables
    n = 2
    m = 200
    mu_pos = [-1, -1]
    mu_neg = [1, 1]
    covariance = [[1, 1], [1, 1]]

    # generate pos and neg points
    np.random.seed(seed)
    A = np.zeros((m, n))
    B = np.zeros((m, n))
    for i in range(0, m):
        A[i] = np.random.normal(mu_pos, covariance, (1, n))
        B[i] = np.random.normal(mu_neg, covariance, (1, n))

    # get X and Y values
    X = np.concatenate([A, B])
    Y = np.zeros((2*m, 1))
    for i in range(0, m):
        Y[i, 0] = 1
        Y[i+m, 0] = -1

    # return sample
    return X, Y