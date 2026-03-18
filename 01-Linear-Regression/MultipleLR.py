import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class MyMultipleLR:
    """Multiple Linear Regression using closed-form (Normal Equation)."""

    def __init__(self):
        self.coef_ = np.array([])
        self.intercept_ = 0.0

    def fit(self, X, y):
        # Add bias column (column of 1s)
        X_b = np.insert(X, 0, 1, axis=1)
        # Normal equation: β = (X^T X)^-1 X^T y
        betas = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_


# Load dataset (10 features, 442 samples)
X, y = load_diabetes(return_X_y=True)
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# Sklearn model
sk_model = LinearRegression()
sk_model.fit(X_train, y_train)
sk_pred = sk_model.predict(X_test)

print("--- Sklearn LinearRegression ---")
print(f"  MSE: {mean_squared_error(y_test, sk_pred):.4f}")
print(f"  MAE: {mean_absolute_error(y_test, sk_pred):.4f}")
print(f"  R²:  {r2_score(y_test, sk_pred):.4f}")

# Custom model
my_model = MyMultipleLR()
my_model.fit(X_train, y_train)
my_pred = my_model.predict(X_test)

print("\n--- Custom MultipleLR (Normal Equation) ---")
print(f"  MSE: {mean_squared_error(y_test, my_pred):.4f}")
print(f"  MAE: {mean_absolute_error(y_test, my_pred):.4f}")
print(f"  R²:  {r2_score(y_test, my_pred):.4f}")
