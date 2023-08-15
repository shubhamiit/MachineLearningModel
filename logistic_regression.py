

class LogisticrRegression:

    def __init__(self, lr =0.01, n_iters =1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # y = wx + b
            predicted = np.dot(X, self.weights) + self.bias

            y_predicted = sigmoid(predicted)
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        predictions = np.dot(X, self.weights) + self.bias

        return [ 1 if sigmoid(predicted)> 0.5 else 0 for predicted in predictions]

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

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)

    clf = LogisticrRegression(lr=.0001, n_iters=1000)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(predictions[:5])
    # print(X.shape)
    print(f"aacuracy is {accuracy(y_test, predictions)}")