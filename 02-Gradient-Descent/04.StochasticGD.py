import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

class StochasticGradientDescent:
    def __init__(self, learning_rate=0.01, epochs=50):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = np.array([])
        self.bias = 0.0
        self.losses: list[float] = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.epochs):
            # Shuffle data each epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            epoch_loss = 0.0
            for i in indices:
                xi = X[i]
                yi = y[i]

                # Predict for single sample
                y_pred = np.dot(xi, self.weights) + self.bias
                error = y_pred - yi

                # Update weights using single sample gradient
                self.weights -= self.lr * error * xi
                self.bias -= self.lr * error

                epoch_loss += error ** 2

            self.losses.append(epoch_loss / n_samples)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# Load dataset
X, y = fetch_california_housing(return_X_y=True)
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (important for gradient descent)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SGD
np.random.seed(42)
model = StochasticGradientDescent(learning_rate=0.01, epochs=50)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Evaluate
print(f"MSE:  {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")

# Plot loss curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(model.losses) + 1), model.losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('SGD: Loss Over Epochs')
plt.grid(True)
plt.tight_layout()
plt.show()
