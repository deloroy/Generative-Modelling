import numpy as np
import math
import matplotlib.pyplot as plt

def add_ones(X):  # to add column of ones to X (in order to learn intercept)
    (n, d) = np.shape(X)
    Xnew = np.ones((n, d + 1))
    Xnew[:, 0:d] = X
    return Xnew

def sig(w, X):  # return 1/(1 + exp(-(wX+b)))
    f = np.vectorize(lambda u: 1/(1 + math.exp(-u)))
    return f(np.dot(X, w))

def plot_points(X,Y): #X 2D points, Y labels
    assert (X.shape[1] == 2)
    idx1 = np.where(Y == 1)[0]
    plt.scatter(X[idx1, 0], X[idx1, 1], color="blue", linewidths = 0.1)
    idx0 = np.where(Y == 0)[0]
    plt.scatter(X[idx0, 0], X[idx0, 1], color="red", linewidths = 0.1)
