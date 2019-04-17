from class_classifier import *
from tools import *


class QuadraticDiscriminantAnalyzer(Classifier):
    def __init__(self):

        self.pi = None
        self.mu1 = None
        self.mu0 = None
        self.inv_sigma0 = None
        self.inv_sigma1 = None

    def fit(self, X, Y):  # fit to train data (X design matrix, Y labels)
        # implement MLE estimator

        (N, d) = np.shape(X)
        Yb = np.sum(Y)  # number of 1 labels
        YXb = np.sum(np.multiply(np.transpose(X), Y), 1) #sum over X for pairs (X,Y) such that Y=1
        Xb = np.sum(X, 0) #sum over X

        self.pi = Yb / N
        self.mu1 = YXb / Yb
        self.mu0 = (Xb - YXb) / (N - Yb)

        X1_centered = X[np.where(Y == 1)[0], :] - self.mu1
        sigma = np.dot(np.transpose(X1_centered), X1_centered) / Yb
        self.inv_sigma1 = np.linalg.inv(sigma)

        X0_centered = X[np.where(Y == 0)[0], :] - self.mu0
        sigma = np.dot(np.transpose(X0_centered), X0_centered) / (N - Yb)
        self.inv_sigma0 = np.linalg.inv(sigma)

    def predict(self, X): # predict the labels for design matrix X
        N = np.shape(X)[0]
        eps = math.pow(10, -10)
        tmp0 = X - self.mu0
        tmp1 = X - self.mu1
        predictions = []
        for i in range(N):
            pred = np.dot(np.dot(tmp0[i], self.inv_sigma0), tmp0[i].T) - np.dot(np.dot(tmp1[i], self.inv_sigma1), tmp1[i].T)
            predictions.append(pred)
        predictions = np.array(predictions)
        predictions += - np.log(np.linalg.det(self.inv_sigma0)+eps) + np.log(np.linalg.det(self.inv_sigma1)+eps) + 2*np.log(self.pi+eps) - 2*np.log(1-self.pi+eps)
        return predictions > 0

    def plot_frontier(self, Xmin, Xmax, ax):  # plot the conic P(Y=1|X) = 0.5 for 2D points

        xs = np.linspace(Xmin[0], Xmax[0], 100)
        ys = np.linspace(Xmin[1], Xmax[1], 100)

        X, Y = np.meshgrid(xs, ys)

        (Nx,Ny) = X.shape
        data = np.zeros((Nx*Ny, 2))
        for i in range(Nx):
            for j in range(Ny):
                idx = i+Nx*j
                data[idx, 0] = X[i,j]
                data[idx, 1] = Y[i,j]

        predictions = self.predict(data)

        Z = np.zeros(np.shape(X))
        for i in range(Nx):
            for j in range(Ny):
                idx = i + Nx*j
                Z[i,j] = predictions[idx]

        ax.contour(X, Y, Z, [0])
