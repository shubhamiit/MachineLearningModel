

class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # init mean, var, priors
        # we will model each feature for each class as gaussian distribution

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes): 
            X_c = X[y==c]
            self._mean[idx,:] = X_c.mean(axis=0)
            self._var[idx,:] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)


    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return y_pred
    
    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        
        return self._classes[np.argmax(posteriors)]
     
    def _pdf(self, class_idx, x):

        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-(x-mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    



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

    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)
    # print(X_train[:5])
    # print(y_train[:5])
    nb= NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    # print(predictions[:5])
    X_c = X[y==0]

    print(X_c.mean(axis=0).shape)
    print(f"aacuracy is {accuracy(y_test, predictions)}")