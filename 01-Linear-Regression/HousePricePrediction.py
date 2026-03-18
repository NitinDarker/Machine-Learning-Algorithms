import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load dataset (California Housing: 20640 samples, 8 features)
# Downloads only on first run, then cached locally
X, y = fetch_california_housing(return_X_y=True)
X = np.array(X)
y = np.array(y)

print(f"Dataset shape: {X.shape}")
print(f"Target: Median house value (in $100,000s)")

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 2. Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"  MSE  (Mean Squared Error):  {mse:.4f}")
print(f"  MAE  (Mean Absolute Error): {mae:.4f}")
print(f"  R² Score:                   {r2:.4f}")

# 3. Plot Predicted vs Actual house prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, edgecolors='k', linewidths=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual House Price ($100,000s)')
plt.ylabel('Predicted House Price ($100,000s)')
plt.title('Predicted vs Actual House Prices')
plt.legend()
plt.tight_layout()
plt.show()
