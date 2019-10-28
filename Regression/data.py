import numpy as np


def gen_data(seed=19719):
    # variables
    n = 2
    m = 200
    mu_pos = np.array([-1, -1])
    mu_neg = np.array([1, 1])
    covariance = np.array([
        [1, 0],
        [0, 1]
    ])

    # generate pos and neg points
    np.random.seed(seed)
    A = np.zeros((m, n))
    B = np.zeros((m, n))
    for i in range(0, m):
        A[i] = np.random.multivariate_normal(mu_pos, covariance)
        B[i] = np.random.multivariate_normal(mu_neg, covariance)

    # get X and Y values
    X = np.concatenate([A, B])
    Y = np.zeros((2*m, 1))
    for i in range(0, m):
        Y[i, 0] = 1
        Y[i+m, 0] = -1

    # return sample
    return X, Y
