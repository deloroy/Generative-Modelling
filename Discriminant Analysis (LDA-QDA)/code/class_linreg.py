from class_classifier import *
from tools import *


class LinearRegression(Classifier):
    def __init__(self):
        self.w = None  # w = (weights intercept) = (w b)

    def fit(self, X, Y):  # fit to train data (X design matrix, Y labels)
        X1 = add_ones(X)
        (N, d) = np.shape(X1)
        XTX = np.dot(np.transpose(X1), X1)
        XTXm1 = np.linalg.inv(XTX)
        XTXm1XT = np.dot(XTXm1, np.transpose(X1))
        self.w = np.dot(XTXm1XT, Y)

    def predict(self, X):  # predict the labels for design matrix X
        X1 = add_ones(X)
        return np.dot(X1, self.w) > 0.5  # P(y=1|x)/P(y=1|y)=sig(self.w,X)

    def plot_frontier(self, Xmin, Xmax, ax=0):  # plot the line P(Y=1|X) = 0.5 for 2D points
        xmin = Xmin[0]
        xmax = Xmax[0]
        ymin = Xmin[1]
        ymax = Xmax[1]
        assert(self.w.shape[0] == 3)
        y1 = - (self.w[2] - 0.5 + self.w[0]*xmin)/self.w[1]
        y2 = - (self.w[2] - 0.5 + self.w[0]*xmax)/self.w[1]
        plt.plot([xmin, xmax], [y1, y2])
        plt.ylim((4 / 3 * ymin, 4 / 3 * ymax))
