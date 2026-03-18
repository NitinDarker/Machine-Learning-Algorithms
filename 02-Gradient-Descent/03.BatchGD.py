import numpy as np

cgpa = np.array([
    5.0, 5.2, 5.5, 5.7, 6.0, 6.2, 6.5, 6.7, 6.8, 7.0,
    7.1, 7.3, 7.5, 7.6, 7.8, 8.0, 8.1, 8.3, 8.5, 8.6,
    8.8, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8
])

placement = np.array([
    2.0, 2.2, 2.5, 2.7, 3.0, 3.2, 3.8, 4.0, 4.2, 4.5,
    4.8, 5.2, 5.8, 6.0, 6.5, 7.0, 7.2, 7.8, 8.5, 9.0,
    9.8, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 15.0
])


class BatchGD:
    """Batch Gradient Descent — uses ALL samples to compute gradient each step."""

    def __init__(self, epochs=1000, lr=0.000001):
        self.epochs = epochs
        self.lr = lr
        self.m = 0.0
        self.b = 0.0

    def train(self, X, y):
        n = len(X)
        for _ in range(self.epochs):
            y_pred = self.m * X + self.b
            gradient_m = (-2 / n) * np.sum(X * (y - y_pred))
            gradient_b = (-2 / n) * np.sum(y - y_pred)
            self.m -= self.lr * gradient_m
            self.b -= self.lr * gradient_b

    def predict(self, X):
        return self.m * X + self.b


model = BatchGD(epochs=1000, lr=0.000001)
model.train(cgpa, placement)
print(f"Slope (m):     {model.m:.4f}")
print(f"Intercept (b): {model.b:.4f}")
print(f"Prediction for CGPA 9.0: {model.predict(9.0):.2f}")
