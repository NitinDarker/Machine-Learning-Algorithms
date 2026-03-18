import math
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class MyGaussianNB:

    def __init__(self):
        self.summaries: dict[int, list[tuple[float, float]]] = {}
        self.priors: dict[int, float] = {}

    def fit(self, X, y):
        separated: dict[int, list] = {}
        for i in range(len(X)):
            label = y[i]
            if label not in separated:
                separated[label] = []
            separated[label].append(X[i])

        total = len(y)
        for label, rows in separated.items():
            self.summaries[label] = []
            for column in zip(*rows):
                mu = sum(column) / len(column)
                var = sum((x - mu) ** 2 for x in column) / len(column)
                self.summaries[label].append((mu, var))
            self.priors[label] = len(rows) / total

    def _gaussian_probability(self, x, mean, var):
        epsilon = 1e-9
        exponent = math.exp(-((x - mean) ** 2) / (2 * var + epsilon))
        return (1 / math.sqrt(2 * math.pi * var + epsilon)) * exponent + epsilon

    def _calculate_posterior(self, row):
        posteriors: dict[int, float] = {}
        for label, features in self.summaries.items():
            log_prob = math.log(self.priors[label])
            for i in range(len(features)):
                mu, var = features[i]
                prob = self._gaussian_probability(row[i], mu, var)
                log_prob += math.log(prob + 1e-9)
            posteriors[label] = log_prob
        return posteriors

    def predict(self, X):
        predictions = []
        for row in X:
            posteriors = self._calculate_posterior(row)
            predictions.append(max(posteriors, key=lambda k: posteriors[k]))
        return predictions


X, y = load_breast_cancer(return_X_y=True)
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MyGaussianNB()
model.fit(X_train.tolist(), y_train.tolist())
y_pred = model.predict(X_test.tolist())

accuracy = sum(1 for a, b in zip(y_pred, y_test) if a == b) / len(y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
