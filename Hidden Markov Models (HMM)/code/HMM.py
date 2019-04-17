from GMM import GMM
from imports import *


#To be consistent with GMM implementation, we kept denoting :
#Z the hidden states (correspond to Q in the report)
#and X the observations (correspond to U in the report)

#Notes :
#question 1. is solved with "log_alpha_msgs", "log_beta_msgs", "compute_proba_Zt_cond_X" and "compute_proba_Zt_and_Znext_cond_X" methods of HMM class
#question 3. is solved with "fit" method of HMM class
#question 4. is solved with "log_alpha_max_msgs" and "decode" methods of HMM class (and "plot_clusters" for visualization)


def log_sum_exp_by_line(X):
    # returns the log of the sums along each line of exp(X)
    # (trick implemented to increase numerical stability for small entries)

    if len(X.shape)==2:
        max = np.max(X, axis=1)
        log_sum = max + np.log(np.sum(np.exp(X - max[:,None]), axis=1))
        return log_sum

    elif len(X.shape)==1: #if X has only one line, summing all elements
        max = np.max(X)
        log_sum = max + np.log(np.sum(np.exp(X - max)))
        return log_sum

    else:
        exit()

class HMM():
    #implements the Hidden Markov Model

    def init(self):
        self.K = None # number of hidden states
        self.A = None # (K,K) matrix of transition probabilities : A_ij : from state i to state j
        self.pis = None  # (K) initial probabilities of each hidden state
        self.mus = None  # (K,D) mean of each gaussian
        self.Sigmas2 = None  # (K,D,D) covariance matrix of each gaussian

    def compute_proba_Xt_cond_Zt(self, Xt):
        # returns P(Xt|Zt=k) for each k, as an array of size (K)
        # Xt is an observation (array of size (D))
        res = np.zeros(self.K)
        for k in range(self.K):
            res[k] = multivariate_normal.pdf(Xt, self.mus[k], self.Sigmas2[k])
        return res

    def log_alpha_msgs(self, t, X):
        # returns the log-alpha messages of the sum-product algorithm
        # from 0 to t, as an array of size (t+1,K)
        # ie. returns the log(alpha_i(Z_i=k)) for each i in [0,t] and each k
        # X is the sequence of observations (array of size (T,D))
        if t==0:
            return np.log(self.pis[None,:])
        else:
            log_alpha_msg_old = self.log_alpha_msgs(t-1, X)

            arr = np.log(self.A.T) + log_alpha_msg_old[-1][None,:]       #arr_ij = log(A_ji * alpha_j)
            tmp = log_sum_exp_by_line(arr)                               #exp(tmp_i) = sum_j exp(arr_ij))

            p_X_cond_Z = self.compute_proba_Xt_cond_Zt(X[t]) # (K)
            new_log_alpha_msg = np.log(p_X_cond_Z) + tmp # (K)           #exp(new_log_alpha_msg(i)) = exp(tmp_i*p(x|z=i))

            return np.vstack([log_alpha_msg_old, new_log_alpha_msg])

    def log_beta_msgs(self, t, X):
        # returns the log-beta messages of the sum-product algorithm
        # from t to T-1, as an array of size (T-t,K)
        # ie. returns the log(beta_i(Z_i=k)) for each i in [t,T-1] and each k
        # X is the sequence of observations (array of size (T,D))

        T = X.shape[0]
        if t==T-1:
            return np.zeros(self.K)[None,:] #np.log(np.ones(self.K)[None,:])
        else:
            log_beta_msg_old = self.log_beta_msgs(t+1, X)
            tmp = log_beta_msg_old[0] + np.log(self.compute_proba_Xt_cond_Zt(X[t+1]))  #(K)

            arr = np.log(self.A) + tmp[None,:]                   #arr_ij = log(A_ij * beta_j * p(x|z=j))
            new_log_beta_msg = log_sum_exp_by_line(arr)          #exp(new_log_beta_msg(i)) = sum_j exp(arr_ij)

            return np.vstack([new_log_beta_msg, log_beta_msg_old])

    def compute_proba_Zt_cond_X(self, X):
        # returns p(z_t = k | x_0 , . . . , x_T ) for all t and each k
        # as an array of size (T, K)
        # X is the sequence of observations (array of size (T,D))
        # uses the sum product algorithm to do so

        T = X.shape[0]
        log_num = self.log_alpha_msgs(T-1,X) + self.log_beta_msgs(0,X) #(T,K)

        log_den = log_sum_exp_by_line(log_num)[:,None] #normalizing to have probabilities

        return np.exp(log_num-log_den)

    def compute_proba_Zt_and_Znext_cond_X(self, X):
        # returns p(z_t = k, z_(t+1) = k' | x_0 , . . . , x_T ) for all t and each k,k'
        # as an array of size (T-1,K,K)
        # if M is the returned array, M[t,k,k'] = p(z_t = k, z_(t+1) = k' | x_0 , . . . , x_T )
        # X is the sequence of observations (array of size (T,D))
        # uses the sum product algorithm to do so

        T = X.shape[0]
        res = -np.ones((T-1,self.K,self.K))

        log_alpha_msgs = self.log_alpha_msgs(T-1,X)
        log_beta_msgs = self.log_beta_msgs(0,X)

        for t in range(T-1):

            log_alpha_m = log_alpha_msgs[t]
            log_beta_m = log_beta_msgs[t+1]
            tmp = self.compute_proba_Xt_cond_Zt(X[t+1])[None,:]  #(1,K)
            log_num = np.log(self.A) + np.add.outer(log_alpha_m,log_beta_m) + np.log(tmp) #exp(log_num(i,j)) = A_ij * alp_i* beta_j * p(x|z=j)

            log_denum = log_sum_exp_by_line(log_num.flatten()) #normalizing to have probabilities
            # in fact, log_denum is the same for all t : equal to p(x_0 ,...,x_T)
            # we could have also computed it with log_sum_exp_by_line(log_beta_msgs+log_alpha_msgs)[0]

            res[t,:,:] = np.exp(log_num-log_denum)

        return res

    def compute_log_likelihood(self, X):
        # returns the log-likelihood of the sequence X given the model
        # it is computed marginalizing over the last hidden state Z_(T-1) (cf. report)

        T = X.shape[0]
        log_alpha_msg = self.log_alpha_msgs(T - 1, X)[T-1]  # alpha_(T-1)*beta_(T-1) = alpha_(T-1)
        return log_sum_exp_by_line(log_alpha_msg)

    def fit(self, X, K, eps=pow(10,-2)):
        # fits the parameters of the HMM using EM algorithm
        # X is the sequence of observations (array of size (T,D)),
        # K is the number of hidden states
        # eps : tolerance on log likelihood difference between two iterations for convergence of EM algorithm

        self.K = K
        T, D = X.shape

        # initialization of means and covariances with GMM
        print("Initialization of Gaussians parameters (means and covariances) with GMM : ")
        gmm_model = GMM(isotropic=False)
        gmm_model.fit(X, K, eps=eps)
        self.mus = gmm_model.mus
        self.Sigmas2 = gmm_model.Sigmas2

        print("\nFit of HMM : ")
        # initialization of pis and A at random
        self.pis = np.random.rand(self.K)
        self.pis /= np.sum(self.pis)
        self.A = np.random.rand(self.K,self.K)
        self.A /= np.sum(self.A,axis=1)[:,None]

        lik = self.compute_log_likelihood(X)
        print("Initial log-likelihood : ", lik)

        delta_lik = 1
        cpt_iter = 1

        while (delta_lik > eps):

            # Expectation step
            pi = self.compute_proba_Zt_cond_X(X)                 # array (T,K) (t,i) -> p(z_t = i|X; θ)
            pij = self.compute_proba_Zt_and_Znext_cond_X(X)      # tensor (T-1,K,K) (t,i,j) -> p(z_(t+1) = j, z t = i|X; θ)

            # Maximization step

            self.pis = pi[0, :]
            pi_repeated = pi[:, :, np.newaxis]  # (T,K,D)
            self.mus = np.sum(pi_repeated * X[:, np.newaxis, :], axis=0) / np.sum(pi_repeated, axis=0)

            self.Sigmas2 = []
            for k in range(self.K):
                Xc = X - self.mus[k]
                Sigmas2k = 0
                for t in range(T):
                    xt = Xc[t, :][:, None]  # size (d,1)
                    Sigmas2k += np.dot(xt, xt.T) * pi[t, k]
                Sigmas2k /= np.sum(pi[:, k])
                self.Sigmas2.append(Sigmas2k)
            self.Sigmas2 = np.array(self.Sigmas2)

            self.A = np.sum(pij,axis=0)/np.sum(pi[:-1],axis=0)[:,None]

            # Computing new likelihood, and deciding if we should stop
            old_lik = lik  # storing old_likelihood to compute delta_lik
            lik = self.compute_log_likelihood(X)  # storing new likelihood
            delta_lik = lik - old_lik  # measure to decide if we should stop or iterate again
            print("Iter " + str(cpt_iter) + " ; log_likelihood : " + str(lik))
            cpt_iter += 1

        print("EM algorithm converged.")

        print("initial distribution found (rounded, 2 decimals) : ", np.round(self.pis,2))
        print("transition matrix found (rounded, 2 decimals) : ", np.round(self.A,2))

    def log_alpha_max_msgs(self, t, X):
        # returns a tuple with two arrays

        # first array is the log-alpha messages for the max-product algorithm from 0 to t
        # ie. the log(alpha_i(Z_i=k)) for each i in [0,t] and each k
        # where alpha_i(Z_i=k)) = p(Y_i |Z_i=k) * max_Z(i-1) [p(Z_i=k|Z_(i-1))*alpha_(i-1)(Z_(i-1))]
        # (array of size (t+1,K))

        # second array contains the state Z_(i-1) argmaximum of p(Z_i=k|Z_(i-1))*alpha_(i-1)(Z_(i-1))
        # for each i in [0,t-1] and each k
        # (array of size (t,K) stored in an array of size(t+1,K) where line 0 is set to -np.ones((1,K)))

        # X is the sequence of observations (array of size (T,D))

        if t==0:
            return np.log(self.pis[None,:]), -np.ones((1,self.K))
        else:
            log_alpha_msg_old, argmax_alpha_max_msgs_old = self.log_alpha_max_msgs(t-1, X)

            arr = np.log(self.A.T) + log_alpha_msg_old[-1][None,:]         #arr_ij = log(A_ji * alpha_j)
            new_argmax_alpha_max_msg = np.argmax(arr,axis=1)               #argmax_j arr_ij
            tmp = arr[np.arange(self.K),new_argmax_alpha_max_msg]          #exp(tmp_i) = max_j exp(arr_ij))
            #same as tmp = np.max(arr,axis=1)

            p_X_cond_Z = self.compute_proba_Xt_cond_Zt(X[t]) # (K)
            new_log_alpha_msg = np.log(p_X_cond_Z) + tmp     # (K)

            return np.vstack([log_alpha_msg_old, new_log_alpha_msg]), \
                   np.vstack([argmax_alpha_max_msgs_old, new_argmax_alpha_max_msg])


    def decode(self, X):
        # returns the sequence of more likely states Z as an array of size T
        # X being the sequence of observations of size T (array of size (T,D))
        # uses the max-product algorithm to do so

        T = X.shape[0]
        log_alpha_max_msgs, argmax_alpha_max_msgs = self.log_alpha_max_msgs(T-1, X)

        sequence_Z = -1 * np.ones(T)
        sequence_Z[T-1] = np.argmax(log_alpha_max_msgs[T-1])
        t = T-2
        while (t>-1):
            sequence_Z[t] = argmax_alpha_max_msgs[t+1,int(sequence_Z[t+1])]
            t-=1

        return sequence_Z

    def plot_covariance_matrices(self, eps, colors, ax):
        #to plot the eps*100 % -mass distribution of the various gaussians

        D = self.mus[0].shape[0]

        for k in range(self.K):

            center = self.mus[k]
            plt.scatter(center[0], center[1], color=colors[self.K], marker='+', linewidths=0.5)

            invFeps = chi2.ppf(eps, df=D)  # reciprocal image of eps by the Chi2 repartition function

            Sigma2 = self.Sigmas2[k] * invFeps   #"normalized" covariance matrix

            eigenval, eigenvect = np.linalg.eig(Sigma2)
            if eigenval[0] > eigenval[1]:
                semiwidth2 = eigenval[0]
                semiheight2 = eigenval[1]
                v = eigenvect[:, 0] #eigenvector associated to largest eigenvalue
            else:
                semiwidth2 = eigenval[1]
                semiheight2 = eigenval[0]
                v = eigenvect[:, 1]  #eigenvector associated to largest eigenvalue
            theta = np.arctan(v[1] / v[0]) # orientation of the ellipse

            e = Ellipse((center[0], center[1]), 2*np.sqrt(semiwidth2), 2*np.sqrt(semiheight2),
                        angle = theta*180/math.pi, edgecolor=colors[k], facecolor='none', linewidth=2)
            ax.add_artist(e)

    def plot_clusters(self, X, title=None, save=False):
        # X is the sequence of observations (array of size (T,D))
        # to plot the assigned hidden states and the eps*100 % -mass distribution of the various gaussians

        assert (X.shape[1] == 2)
        Z = self.decode(X)
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
