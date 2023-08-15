
class KNN:
    def __init__(self, k =3) -> None:
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return predicted_labels
    
    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = sort_indices(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]  
        # return the most common class label
        most_common = Counter(k_neighbor_labels)
        return most_common
    
    def euclidean_distance(self, point1, point2):


        squared_distance = sum((a - b)**2 for a, b in zip(point1, point2))
        distance = squared_distance ** 0.5
        return distance
    
def sort_indices(distances):
    return sorted(range(len(distances)),key=lambda k: distances[k])


def Counter( labels):
    class_counter = {}
    for label in labels:
        if label not in class_counter:
            class_counter[label] = 0
        class_counter[label] += 1
    # return key with max value
    return max(class_counter, key=class_counter.get)


print(sort_indices([5,2,4,1,6]))
