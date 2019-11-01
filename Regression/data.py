import numpy as np


def gen_data(seed=19719):
    # variables
    n = 2  # features
    m = 200  # sample size
    mu_pos = np.array([-1, -1])  # + mean
    mu_neg = np.array([1, 1])  # - mean
    covariance = np.array([
        [1, 0],
        [0, 1]
    ])

    # generate pos and neg points
    r = np.random.RandomState(seed)
    A = np.zeros((m, n))  # + features
    B = np.zeros((m, n))  # - features
    for i in range(0, m):
        A[i] = r.multivariate_normal(mu_pos, covariance)
        B[i] = r.multivariate_normal(mu_neg, covariance)

    # get X and Y values
    X = np.concatenate([A, B])  # concatenate + and - features
    Y = np.zeros((2*m, 1))
    for i in range(0, m):  # fill Y values with 1 and -1
        Y[i, 0] = 1
        Y[i+m, 0] = -1

    # return sample
    return X, Y