import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class SimpleNN:
    def __init__(self, input_size, hidden_size, lr=0.01, epochs=5000):
        self.lr = lr
        self.epochs = epochs

        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        n = X.shape[0]

        for _ in range(self.epochs):

            # Forward
            Z1 = X @ self.W1 + self.b1
            A1 = self.sigmoid(Z1)

            Z2 = A1 @ self.W2 + self.b2
            A2 = self.sigmoid(Z2)

            # Backprop
            dZ2 = A2 - y
            dW2 = (1 / n) * (A1.T @ dZ2)
            db2 = (1 / n) * np.sum(dZ2, axis=0, keepdims=True)

            dZ1 = (dZ2 @ self.W2.T) * A1 * (1 - A1)
            dW1 = (1 / n) * (X.T @ dZ1)
            db1 = (1 / n) * np.sum(dZ1, axis=0, keepdims=True)

            # Update
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

    def predict(self, X):
        A1 = self.sigmoid(X @ self.W1 + self.b1)
        A2 = self.sigmoid(A1 @ self.W2 + self.b2)

        return (A2 >= 0.5).astype(int).flatten()


X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SimpleNN(
    input_size=X_train.shape[1],
    hidden_size=16
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))