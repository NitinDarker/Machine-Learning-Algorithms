import math
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report


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


# Load dataset
X, y = load_breast_cancer(return_X_y=True)
X = np.array(X)
y = np.array(y)
target_names = ["malignant", "benign"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Custom Naive Bayes
custom_model = MyGaussianNB()
custom_model.fit(X_train.tolist(), y_train.tolist())
custom_pred = custom_model.predict(X_test.tolist())
custom_acc = accuracy_score(y_test, custom_pred)

# Sklearn Naive Bayes
sk_model = GaussianNB()
sk_model.fit(X_train, y_train)
sk_pred = sk_model.predict(X_test)
sk_acc = accuracy_score(y_test, sk_pred)

# Results
print("--- Custom Gaussian NB ---")
print(f"Accuracy: {custom_acc:.4f}")
print(classification_report(y_test, custom_pred, target_names=target_names))

print("--- Sklearn Gaussian NB ---")
print(f"Accuracy: {sk_acc:.4f}")
print(classification_report(y_test, sk_pred, target_names=target_names))
