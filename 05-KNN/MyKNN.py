import numpy as np
import pandas as pd
from collections import Counter

class MyKNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = np.array([])
        self.y_train = np.array([])

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X_test):
        predictions = []

        for test_point in X_test:
            distances = []

            # Calculate distance from all training points
            for idx, train_point in enumerate(self.X_train):
                dist = self._euclidean_distance(test_point, train_point)
                distances.append((idx, dist))

            # Sort by distance
            distances.sort(key=lambda x: x[1])

            # Take k nearest neighbors
            neighbors = distances[:self.k]

            # Majority voting
            neighbor_labels = [
                self.y_train[idx] for idx, _ in neighbors
            ]
            most_common = Counter(neighbor_labels).most_common(1)[0][0]
            predictions.append(most_common)

        return np.array(predictions)

def split_data(X,y,test_size=0.25,random_state=42):#25% data for testing,random_state=42 → ensures same random split every time
    np.random.seed(random_state) # Every time you run the program → same train/test split
    indices = np.arange(len(X)) # [0,1,2,....len(X)-1]
    np.random.shuffle(indices) # [127, 32, 210, 5, 89, ...]

    test_len = int(len(X) * test_size) # 100*.25 => 25
    test_idx = indices[:test_len] # random first 25 index
    train_idx = indices[test_len:] # 26 .... 100

    # print(test_idx)
    # print(train_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def standard_scaler(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    # Avoid division by zero
    std[std == 0] = 1

    X_scaled = (X - mean) / std
    return X_scaled, mean, std

df = pd.read_csv("D:/Code/ML/Machine-Learning-Algorithms/05-KNN/Social_Network_Ads.csv")

# Drop User ID
df.drop("User ID", axis=1, inplace=True)

# Encode Gender
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

X = df[["Gender","Age", "EstimatedSalary"]].values
y = df["Purchased"].values

# Split data
X_train, X_test, y_train, y_test = split_data(X, y)

# Scale training data
X_train, mean, std = standard_scaler(X_train)

# Scale test data using training mean/std
X_test = (X_test - mean) / std

# Create KNN model
knn = MyKNN(k=5)

# Train
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Accuracy
accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Accuracy:", accuracy)