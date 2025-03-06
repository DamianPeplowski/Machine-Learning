from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

# X_train: Feature set for training data.
# X_test: Feature set for test data.
# y_train: Class labels for training data.
# y_test: Class labels for test data.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def cosine_distance(x1, x2):
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    return 1 - (dot_product / (norm_x1 * norm_x2))

def knn_function(k, X_train, y_train, X_test, distance_metric):
    predicted_classes = []

    for x_test in X_test:
        test_distances = []
        for x_train in X_train:
            if distance_metric == "euclidean":
                test_distances.append(euclidean_distance(x_test, x_train))
            elif distance_metric == "manhattan":
                test_distances.append(manhattan_distance(x_test, x_train))
            elif distance_metric == "cosine":
                test_distances.append(cosine_distance(x_test, x_train))

        k_nearest_neighbors_indices = np.argsort(test_distances)[:k]

        k_nearest_neighbors = []
        for i in k_nearest_neighbors_indices:
            k_nearest_neighbors.append(y_train[i])

        class_counts = {}
        for cls in k_nearest_neighbors:
            if cls in class_counts:
                class_counts[cls] += 1
            else:
                class_counts[cls] = 1

        most_common_class = None
        max_count = 0
        for key, count in class_counts.items():
            if count > max_count:
                most_common_class = key
                max_count = count

        predicted_classes.append(most_common_class)

    return predicted_classes

def accuracy_score(y_true, predicted_classes):
    return np.sum(y_true == predicted_classes) / len(y_true)

k = 15
metrics = ["euclidean", "manhattan", "cosine"]

accuracies = []

for metric in metrics:
    predicted_classes = knn_function(k, X_train, y_train, X_test, metric)
    accuracy = accuracy_score(y_test, predicted_classes)
    accuracies.append(accuracy)
    print(f'Accuracy for metric {metric}: {accuracy:.2f}')

from sklearn.neighbors import KNeighborsClassifier

for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    print(f'Accuracy of KNN classifier with {metric} metric (scikit-learn): {accuracy:.2f}')
