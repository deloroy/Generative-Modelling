from imports import *
from GMM import GMM
from HMM import HMM

def main():
    pref_path = os.getcwd() + "/classification_data_HWK2/EMGaussian"

    train_data = np.loadtxt(open(pref_path + ".data", "rb"), delimiter=" ")
    test_data = np.loadtxt(open(pref_path + ".test", "rb"), delimiter=" ")

    Xtrain = train_data[:, :2]
    Xtest = test_data[:, :2]

    models = {"GMM":GMM(isotropic=False), "HMM":HMM()}
    K = 4  #number of clusters

    for name in ["GMM","HMM"]:

        print(name)
        model = models[name]
        model.fit(Xtrain, K, eps=pow(10,-2))

        # visualize clusters and frontiers
        model.plot_clusters(Xtrain, "figs/" + name + " on train",save=True)
        model.plot_clusters(Xtest, "figs/" + name + " on test",save=True)

        print("")

        lik = model.compute_log_likelihood(Xtrain)
        print("mean log-likelihood on training set : ", lik/ Xtrain.shape[0])

        lik = model.compute_log_likelihood(Xtest)
        print("mean log-likelihood on test set : ", lik / Xtest.shape[0])

        print("\n------------------------\n")

main()
