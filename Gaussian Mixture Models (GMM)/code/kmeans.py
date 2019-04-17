import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as sp
from matplotlib import colors as mcolors


class Kmeans():
    def __init__(self):
        self.mu = None
        self.z = None

        self.K = None

    def distortion_J(self, X, zv=None):

        if zv is None:
            zv = self.z
        J = 0
        for k in range(self.K):
            J += np.dot(np.transpose(zv[:, k]), np.linalg.norm(X - self.mu[k, :], axis=1))
        return J

    def assign_Z(self, X):
        #return the clusters assigned to X, in a matrix (X.shape[0],K) for K clusters (hot encoding)
        num_data = X.shape[0]

        distances = sp.distance_matrix(X, self.mu)
        z_index = np.argmin(distances, axis=1)

        z = np.zeros((num_data, self.K))
        z[range(num_data), z_index] = 1
        return z.astype(int)

    def update_mu(self, X):
        norm = np.sum(self.z, axis=0)
        norm[norm <= 10 ** -15] = 1
        self.mu = np.dot(np.transpose(self.z), X) / norm.reshape(-1, 1)

    def fit(self, X, K, max_step=1000, prec=10**-10, verbose=True):
        self.K = K

        min_x, min_y = np.min(X, axis=0)
        max_x, max_y = np.max(X, axis=0)

        mu0_x = np.random.uniform(min_x, max_x, self.K)
        mu0_y = np.random.uniform(min_y, max_y, self.K)
        self.mu = np.transpose(np.vstack([mu0_x, mu0_y]))

        self.z = self.assign_Z(X)
        J = self.distortion_J(X)
        J0 = -1

        i = 0
        while abs(J - J0) > prec and i < max_step:
            J0 = J
            self.z = self.assign_Z(X)
            self.update_mu(X)
            J = self.distortion_J(X)
            i += 1
            if verbose == "Full" and i % 100 == 0:
                print(i)

        if verbose:
            print("Kmeans converged in {} iterations.\nFinal value of distortion : {}.\nFinal values of mus :\n{}".format(i, J, self.mu))

    def plot_clusters(self, X, title="", save=False): #X 2D points, Y labels
        assert (X.shape[1] == 2)

        z_plt = self.assign_Z(X)

        if self.K < 5:
            colors = ['y', 'k', 'g', 'b', 'r']
        else:
            colors = list(mcolors.CSS4_COLORS.values())

        fig, ax = plt.subplots()
        for k in range(self.K):
            Xk = X[np.where(z_plt[:, k] == 1)[0], :]
            plt.scatter(Xk[:, 0], Xk[:, 1], color=colors[k], linewidths=0.1)

        plt.scatter(self.mu[:, 0], self.mu[:, 1], color=colors[self.K], marker='+', linewidths=0.5)

        plt.title(title)

        if save :
            plt.savefig(title + ".eps")
        else:
            plt.show()
