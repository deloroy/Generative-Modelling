from class_classifier import *
from tools import *


class LinearDiscriminantAnalyzer(Classifier):
    def __init__(self):

        self.pi = None
        self.mu1 = None
        self.mu0 = None
        self.inv_sigma = None  # inverse of covariance matrix

        self.a = None         # if np.dot(x, b) + a > 0, x belongs to class 1
        self.b = None

    def fit(self, X, Y):  # fit to train data (X design matrix, Y labels)
        # implement MLE estimator

        (n, d) = np.shape(X)
        n1 = np.sum(Y == 1)  # number of 1 labels
        self.pi = n1 / n
        X1 = X[np.where(Y == 1)[0], :]  # data with label 1
        X0 = X[np.where(Y == 0)[0], :]  # data with label 0
        self.mu1 = np.mean(X1, axis=0)
        self.mu0 = np.mean(X0, axis=0)

        sigma = np.zeros((d, d))
        X0_Centered = X0 - self.mu0
        X1_Centered = X1 - self.mu1
        sigma += np.dot(X0_Centered.T, X0_Centered)  # size (d,d)
        sigma += np.dot(X1_Centered.T, X1_Centered)
        sigma /= n
        self.inv_sigma = np.linalg.inv(sigma)

    def predict(self, X): # predict the labels for design matrix X
        eps = math.pow(10, -10)
        self.b = np.dot(self.inv_sigma, self.mu1 - self.mu0)
        self.a = np.log(self.pi + eps) - np.log(1 - self.pi + eps) - 0.5 * np.dot((self.mu1 + self.mu0).T, self.b)
        return (np.dot(X, self.b) + self.a) > 0

    def plot_frontier(self, Xmin, Xmax, ax=0):  # plot the line P(Y=1|X) = 0.5 for 2D points
        xmin = Xmin[0]
        xmax = Xmax[0]
        ymin = Xmin[1]
        ymax = Xmax[1]
        assert(self.b.shape[0] == 2)
        y1 = - (self.a + self.b[0]*xmin)/self.b[1]
        y2 = - (self.a + self.b[0]*xmax)/self.b[1]
        plt.plot([xmin, xmax], [y1, y2])
        plt.ylim((4 / 3 * ymin, 4 / 3 * ymax))
