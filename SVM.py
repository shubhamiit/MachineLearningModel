

class SVM:

    def __init__(self, learning_rate=.001, lambda_param=.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_output = np.dot(X, self.weights) - self.bias

            outputMultiply = np.dot(y_ , linear_output)-1
            if outputMultiply >=0:
                self.weights -= self.lr * (2 * self.lambda_param * self.weights)
            else:
                self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(X.T, y_))
                self.bias -= self.lr * np.sum(y_)

    def  predict(self, X):
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted)**2)




def accuracy(y_true, y_predicted):
    return np.sum(y_true == y_predicted) / len(y_true)
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    y = np.where(y == 0, -1, 1)

    clf = SVM()
    clf.fit(X, y)
    predictions = clf.predict(X)

    print("Accuracy:", accuracy(y, predictions))

