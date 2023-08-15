

class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y_):
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        # y_ = np.array([1 if i > 0 else 0 for i in y])

        # gradient descent
        for _ in range(self.n_iters):

            linear_output = np.dot(X, self.weights) + self.bias
            y_predicted = self.activation_func(linear_output)


            # Perceptron update rule
            update = self.lr * (y_ - y_predicted)
            self.weights += np.dot(X.T, update)
            self.bias += np.sum(update)


            # for idx, x_i in enumerate(X):
            #     linear_output = np.dot(x_i, self.weights) + self.bias
            #     y_predicted = self.activation_func(linear_output)

            #     # Perceptron update rule
            #     update = self.lr * (y_[idx] - y_predicted)
            #     self.weights += update * x_i
            #     self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)

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

    X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)

    p = Perceptron(learning_rate=.01, n_iters=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)
    print(predictions[:5])
    print(X.shape)
    print(y[:5])
    print(f"aacuracy is {accuracy(y_test, predictions)}")
    print(y.shape)
