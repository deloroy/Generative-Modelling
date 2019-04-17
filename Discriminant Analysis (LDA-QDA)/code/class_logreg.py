from class_classifier import *
from tools import *


class LogisticRegression(Classifier):
    def __init__(self):
        self.w = None  # w = (weights intercept) = (w b)

    def fit(self, X, Y):  # fit to train data (X design matrix, Y labels)
        X1 = add_ones(X)
        (n, d) = np.shape(X1)
        # use Newton-Raphson algorithm
        old_w = np.ones(d)
        self.w = np.zeros(d)  # initialization
        tol = math.pow(10, -3)  # tolerance on norm2(w(t)-w(t-1))/norm2(w(t-1)) for convergence
        iter = 0
        while np.linalg.norm(self.w - old_w) > tol * np.linalg.norm(old_w):
            tmp = sig(self.w, X1)  # size n
            grad = np.dot(X1.T, Y - tmp)  # size d
            D = np.diag(tmp * (1 - tmp))  # size (n,n)
            hess = - np.dot(np.dot(X1.T, D), X1)  # size (d,d)
            old_w = self.w.copy()
            eps = math.pow(10, -8) # (adding eps*I to the hessian for the potential case where hess is not invertible)
            self.w -= 1/n * np.dot(np.linalg.inv(hess+eps*np.eye(d)), grad).flatten()
            #print("iter : ", iter,"; gradient diff : ",np.linalg.norm(self.w - old_w)/np.linalg.norm(old_w))
            iter += 1

    def predict(self, X):  # predict the labels for design matrix X
        X1 = add_ones(X)
        return np.dot(X1, self.w) > 0  # P(x=1|y)/P(x=0|y)=sig(self.w,X)

    def plot_frontier(self, Xmin, Xmax, ax=0):  # plot the line P(Y=1|X) = 0.5 for 2D points
        xmin = Xmin[0]
        xmax = Xmax[0]
        ymin = Xmin[1]
        ymax = Xmax[1]
        assert(self.w.shape[0] == 3)
        y1 = - (self.w[2] + self.w[0]*xmin)/self.w[1]
        y2 = - (self.w[2] + self.w[0]*xmax)/self.w[1]
        plt.plot([xmin, xmax], [y1, y2])
        plt.ylim((4 / 3 * ymin, 4 / 3 * ymax))
