import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import colors as mcolors

from scipy.stats import multivariate_normal
from scipy.stats import chi2

from kmeans import Kmeans


def compute_hot_encoding(X, K):
    #X vector with values in [0,K-1]
    #return a matrix of size (N,K) with N=X.shape[0], which stores the hot encoding of X
    res = np.zeros((X.shape[0], K))
    for k in range(K):
        res[:,k] = X == k
    return res


class GMM():
    # implements the Gaussian Mixture Model
    # setting the boolean parameter "isotropic", one can choose between an isotropic or a general model

    def __init__(self, isotropic=False):
        # if isotropic=False, Sigmas2 is an array of dimension K storing the variances for each cluster
        # if isotropic=True, Sigmas2 is an array of dimension (K,D,D) storing the covariance matrices for each cluster
        self.isotropic = isotropic
        self.K = None
        self.pis = None      #(K,)
        self.mus = None      #(K,D)
        self.Sigmas2 = None  #(K) or (K,D,D)

    def compute_proba_X_cond_Z(self, X): #returns an array of size (N,K) returns P(X|Z=k)
        N = X.shape[0]
        proba_X_cond_Z = np.zeros((N, self.K))
        for k in range(self.K):
            proba_X_cond_Z[:, k] = multivariate_normal.pdf(X, self.mus[k], self.Sigmas2[k])
        return proba_X_cond_Z

    def compute_proba_Z_cond_X(self, X): #returns an array of size (N,K) storing P(Z=k|X)
        res = self.compute_proba_X_cond_Z(X)
        for k in range(self.K):
            res[:,k] *= self.pis[k]
        res /= np.sum(res,axis=1)[:,np.newaxis]
        return res

    def compute_log_likelihood(self, X):
        tmp = self.compute_proba_X_cond_Z(X)
        for k in range(self.K):
            tmp[:,k] *= self.pis[k]
        proba_X = np.sum(tmp,axis=1)
        return np.sum(np.log(proba_X))

    def fit(self, X, K, eps=pow(10,-2)):
        #fit the GMM with EM algorithm
        #X data, K number of gaussians
        #eps : tolerance on likelihood difference between two iterations for convergence of EM algorithm
        #if init="KMeans" or "random"

        self.K = K
        N , D = X.shape

        #initialization with KMeans
        kmeans = Kmeans()
        kmeans.fit(X, K, verbose=False)
        proba_Z = kmeans.assign_Z(X)

        #Implementation note :
        #We do M step before E step in order to initialize more easily with Kmeans

        delta_lik = 1
        cpt_iter = 0

        while (delta_lik>eps):

            #Maximization step
            self.pis = np.sum(proba_Z,axis=0)/N
            proba_Z_repeated = proba_Z[:,:,np.newaxis] #(N,K,D)
            self.mus = np.sum(proba_Z_repeated*X[:,np.newaxis,:],axis=0)/np.sum(proba_Z_repeated,axis=0)

            self.Sigmas2 = []
            for k in range(self.K):
                if self.isotropic:
                    dists = np.linalg.norm(X - self.mus[k], axis=1)
                    Sigmas2k = np.sum(dists * dists * proba_Z[:,k], axis=0)
                    Sigmas2k /= np.sum(proba_Z[:, k], axis=0)
                    Sigmas2k /= D
                    self.Sigmas2.append(Sigmas2k)
                else:
                    Xc = X - self.mus[k]
                    Sigmas2k = 0
                    for i in range(N):
                        xi = Xc[i, :][:,None] #size (d,1)
                        Sigmas2k += np.dot(xi, xi.T) * proba_Z[i, k]
                    Sigmas2k /= np.sum(proba_Z[:,k])
                    self.Sigmas2.append(Sigmas2k)
            self.Sigmas2 = np.array(self.Sigmas2)

            # Expectation step
            proba_Z = self.compute_proba_Z_cond_X(X)

            # Computing new likelihood, and deciding if we should stop
            if cpt_iter == 0:
                lik = self.compute_log_likelihood(X)  # storing new likelihood
                delta_lik = 1  # to iterate
                print("Initial likelihood : ", lik)
            else:
                old_lik = lik  # storing old_likelihood to compute delta_lik
                lik = self.compute_log_likelihood(X)  # storing new likelihood
                delta_lik = lik - old_lik  # measure to decide if we should stop or iterate again
                print("Iter " + str(cpt_iter) + " ; delta_likelihood : " + str(delta_lik))
            cpt_iter += 1

        print("EM algorithm converged.")

    def assign_Z(self, X, hot_encoding=True): #return argmax_k P(Z=k|X)
        proba_Z = self.compute_proba_Z_cond_X(X)
        Z_non_vectorized = np.argmax(proba_Z,axis=1)
        if hot_encoding:
            return compute_hot_encoding(Z_non_vectorized, self.K)
        else:
            return Z_non_vectorized

    def plot_covariance_matrices(self, eps, colors, ax): #to plot the eps*100 % -mass distribution of the various gaussians

        D = self.mus[0].shape[0]

        for k in range(self.K):

            center = self.mus[k]
            plt.scatter(center[0], center[1], color=colors[self.K], marker='+', linewidths=0.5)

            invFeps = chi2.ppf(eps, df=D)  # reciprocal image of eps by the Chi2 repartition function
            if self.isotropic:
                radius = math.sqrt(invFeps * self.Sigmas2[k])
                c = plt.Circle((center[0], center[1]), radius,
                               edgecolor=colors[k], facecolor='none', linewidth=2)
                ax.add_artist(c)
            else:
                Sigma2 = self.Sigmas2[k] * invFeps   #"normalized" covariance matrix

                eigenval, eigenvect = np.linalg.eig(Sigma2)
                if eigenval[0] > eigenval[1]:
                    semiwidth2 = eigenval[0]
                    semiheight2 = eigenval[1]
                    v = eigenvect[:, 0] #eigenvector associated to largest eignevalue
                else:
                    semiwidth2 = eigenval[1]
                    semiheight2 = eigenval[0]
                    v = eigenvect[:, 1]  #eigenvector associated to largest eignevalue
                theta = np.arctan(v[1] / v[0]) # orientation of the ellipse

                e = Ellipse((center[0], center[1]), 2*np.sqrt(semiwidth2), 2*np.sqrt(semiheight2),
                            angle = theta*180/math.pi, edgecolor=colors[k], facecolor='none', linewidth=2)
                ax.add_artist(e)

    def plot_clusters(self, X, title=None, save=False):
        assert (X.shape[1] == 2)
        Z = self.assign_Z(X,hot_encoding=False)
        fig, ax = plt.subplots()

        #plot points and labels
        if self.K < 5:
            colors = ['y', 'k', 'g', 'b', 'r']
        else:
            colors = list(mcolors.CSS4_COLORS.values())
        for k in range(self.K):
            Xk = X[np.where(Z==k)[0], :]
            plt.scatter(Xk[:, 0], Xk[:, 1], color=colors[k], linewidths=0.1)

        #plot covariance matrices as circles/ellipsoids containg 0.9 of the mass of the various Gaussian distributions
        self.plot_covariance_matrices(0.9, colors, ax)

        plt.title(title)
        if save:
            plt.savefig(title + ".eps")
        else:
            plt.show()

