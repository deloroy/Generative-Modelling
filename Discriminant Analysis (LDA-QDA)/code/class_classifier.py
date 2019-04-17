from tools import *

class Classifier():

    def score(self, X, Y): # X design matrix, Y true labels
        return np.sum(self.predict(X) == Y) / np.shape(Y)

    def error_rate(self, X, Y):
        return 1 - self.score(X, Y)

    def plot_predictions(self, X, title=None, save=False): #X design matrix
        assert (X.shape[1] == 2)
        predictions = self.predict(X)
        fig, ax = plt.subplots()
        plot_points(X, predictions)
        self.plot_frontier((np.min(X[:, 0]), np.min(X[:, 1])), (np.max(X[:, 0]), np.max(X[:, 1])), ax)
        # plt.gca().set_position([-0.05, -0.05, 1.05, 1.])
        plt.title(title)
        if save :
            plt.savefig(title + ".eps")
        else:
            plt.show()
