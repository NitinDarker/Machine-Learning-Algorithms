import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
X, y = fetch_california_housing(return_X_y=True)
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (important for Ridge)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

# Ridge Regression (alpha controls regularization strength)
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_scaled, y_train)
ridge_pred = ridge.predict(X_test_scaled)

# Evaluate both
print("--- Linear Regression ---")
print(f"  MSE:  {mean_squared_error(y_test, lr_pred):.4f}")
print(f"  MAE:  {mean_absolute_error(y_test, lr_pred):.4f}")
print(f"  R²:   {r2_score(y_test, lr_pred):.4f}")

print("\n--- Ridge Regression (alpha=1.0) ---")
print(f"  MSE:  {mean_squared_error(y_test, ridge_pred):.4f}")
print(f"  MAE:  {mean_absolute_error(y_test, ridge_pred):.4f}")
print(f"  R²:   {r2_score(y_test, ridge_pred):.4f}")

# Compare coefficients
print("\n--- Coefficient Comparison ---")
feature_names = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]
print(f"{'Feature':<12} {'Linear':>10} {'Ridge':>10}")
for name, lc, rc in zip(feature_names, lr.coef_, ridge.coef_):
    print(f"{name:<12} {lc:>10.4f} {rc:>10.4f}")

# Plot: Predicted vs Actual for both
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_test, lr_pred, alpha=0.3, edgecolors='k', linewidths=0.5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title(f'Linear Regression (R²={r2_score(y_test, lr_pred):.4f})')

axes[1].scatter(y_test, ridge_pred, alpha=0.3, edgecolors='k', linewidths=0.5, color='coral')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1].set_xlabel('Actual')
axes[1].set_ylabel('Predicted')
axes[1].set_title(f'Ridge Regression (R²={r2_score(y_test, ridge_pred):.4f})')

plt.tight_layout()
plt.show()
