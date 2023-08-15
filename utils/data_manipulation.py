import numpy as np
def euclidean_distance2(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def euclidean_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimensions")

    squared_distance = sum((a - b)**2 for a, b in zip(point1, point2))
    distance = squared_distance ** 0.5
    return distance

# Example usage with numpy arrays
import numpy as np
train_data = np.array([[-2.6, 1.9, 2.0, 1.0, 1.0], [-2.8, 1.7, -1.2, 1.5, 2.0], [2.0, -0.9, 0.3, 2.3, 0.0], [-1.5, -0.1, -1.6, -1.1, 0.0], [-1.0, -0.6, -1.2, -0.7, 0.0], [-0.3, 1.2, 2.6, 0.2, 1.0], [-1.8, -1.3, -0.1, -1.2, 0.0], [0.2, 1.2, -0.6, -1.3, 1.0], [-5.2, 0.3, 0.2, 2.2, 2.0], [-0.8, -0.1, 1.5, -0.1, 0.0], [-2.3, 0.3, 0.8, 0.7, 2.0], [0.2, 3.0, 3.6, -0.9, 1.0], [1.7, -0.8, -0.0, 2.0, 0.0], [2.8, 0.8, 1.8, -0.7, 2.0]])

# print(train_data[0][0:-1])
# # train_data = np.array([[-2.6, 1.9, 2.0 , 1.0, 1.0],[-2.8, 1.7, -1.2, 1.5, 2.0],[2.0, -0.9, 0.3, 2.3, 0.0],[-1.5, -0.1, -1.6, -1.1, 0.0],[-1.0, -0.6, -1.2, -0.7, 0.0],[-0.3, 1.2, 2.6, 0.2, 1.0],[-1.8, -1.3, -0.1, -1.2, 0.0],[0.2, 1.2, -0.6, -1.3, 1.0],[-5.2, 0.3, 0.2, 2.2, 2.0],[-0.8, -0.1, 1.5, -0.1, 0.0],[-2.3, 0.3, 0.8, 0.7, 2.0],[0.2, 3.0, 3.6, -0.9, 1.0],[1.7, -0.8, -0.0, 2.0, 0.0],[2.8, 0.8, 1.8, -0.7, 2.0]])
# point_a = np.array([1, 2, 3])
# point_b = np.array([4, 5, 6])
distance = euclidean_distance2(train_data[0][0:-1], train_data[1][0:-1])
print("Euclidean distance:", distance)