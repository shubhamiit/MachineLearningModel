

class LinearRegression:

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
            y_predicted = np.dot(X, self.weights) + self.bias

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted)**2)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)

    clf = LinearRegression(lr=.05, n_iters=1000)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(predictions[:5])
    # print(X.shape)
    print(f"mse is {mse(y_test, predictions)}")

    # print(y.shape)

    # # print(y[:5])

    # print(X[:5])

    # # figure = plt.figure(figsize=(8, 6))
    # # plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
    # # plt.show()
    # n_samples, n_features = X.shape
    # weights = np.zeros(n_features)
    # bias = 2

    # y_predicted = np.dot(X, weights) + bias
    # print(y_predicted.shape)
    # print(y_predicted[:5])
    # print(weights.shape)
    # print(weights[:5])

    # dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
    # db = (1 / n_samples) * np.sum(y_predicted - y)

    # print(dw.shape)
    # print(dw[:5])
    # print(db.shape)
    # print(db)

    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color="blue", s=10)
    m2 = plt.scatter(X_test, y_test, color="red", s=10)
    plt.plot(X_test, predictions, color="black", linewidth=2, label="Prediction")
    plt.show()