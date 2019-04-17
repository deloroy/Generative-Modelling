import os
import numpy as np

from kmeans import Kmeans
from GMM import GMM



def main():
    pref_path = os.getcwd() + "/classification_data_HWK2/EMGaussian"

    train_data = np.loadtxt(open(pref_path + ".data", "rb"), delimiter=" ")
    test_data = np.loadtxt(open(pref_path + ".test", "rb"), delimiter=" ")

    Xtrain = train_data[:, :2]
    Xtest = test_data[:, :2]

    models = {"Kmeans":Kmeans(),"GMM_general":GMM(isotropic=False),"GMM_isotropic":GMM(isotropic=True)}
    K=4  #number of clusters

    for name in ["Kmeans","GMM_isotropic","GMM_general"]:

        print(name)
        model = models[name]
        model.fit(Xtrain, 4)

        # visualize clusters and frontiers
        model.plot_clusters(Xtrain, name + " on train",save=False)
        model.plot_clusters(Xtest, name + " on test",save=False)

        if name in ["GMM_general","GMM_isotropic"]:

            lik = model.compute_log_likelihood(Xtrain)
            print("mean log-likelihood on training set : ", lik/ Xtrain.shape[0])

            lik = model.compute_log_likelihood(Xtest)
            print("mean log-likelihood on test set : ", lik / Xtest.shape[0])

        print("")

main()
