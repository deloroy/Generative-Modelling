import os

from class_logreg import LogisticRegression
from class_linreg import LinearRegression
from class_lda import LinearDiscriminantAnalyzer
from class_qda import QuadraticDiscriminantAnalyzer
from tools import *

def main():
    pref_path = os.getcwd() + "/classification_data_HWK1/classification"
    sets = ["A", "B", "C"]

    selected_sets = sets

    for set in selected_sets:

        train_data = np.loadtxt(open(pref_path + set + ".train", "rb"), delimiter="\t")
        test_data = np.loadtxt(open(pref_path + set + ".test", "rb"), delimiter="\t")
        Xtrain = train_data[:, :2]
        Ytrain = train_data[:, 2]
        Xtest = test_data[:, :2]
        Ytest = test_data[:, 2]

        print("Set " + set)
        plot_points(Xtrain,Ytrain)
        plt.title("Set "+str(set)+" train data ; true labels")
        plt.show()
        plot_points(Xtest, Ytest)
        plt.title("Set "+str(set)+" test data ; true labels")
        plt.show()

        LoR = LogisticRegression()
        LiR = LinearRegression()
        LDA = LinearDiscriminantAnalyzer()
        QDA = QuadraticDiscriminantAnalyzer()

        classifiers = {"Logistic Regression":LoR, "Linear Regression": LiR, "LDA":LDA, "QDA": QDA}

        for name in classifiers.keys():
            C = classifiers[name]
            C.fit(Xtrain, Ytrain)
            C.plot_predictions(Xtrain, "train_{}_{}".format(set,name), False)
            print("Misclassication error for " + name + " model on train data : ", C.error_rate(Xtrain, Ytrain))
            C.plot_predictions(Xtest, "test_{}_{}".format(set,name), False)
            print("Misclassication error for " + name + " model on test data : ", C.error_rate(Xtest, Ytest))
            print(" ")

main()
