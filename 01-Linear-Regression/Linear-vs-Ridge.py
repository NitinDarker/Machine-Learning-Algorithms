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

print("--- Linear Regression ---")
print(f"  MSE:  {mean_squared_error(y_test, lr_pred):.4f}")
print(f"  MAE:  {mean_absolute_error(y_test, lr_pred):.4f}")
print(f"  R²:   {r2_score(y_test, lr_pred):.4f}")

# Ridge Regression with different alpha values
alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
ridge_results: list[tuple[float, float, float, float]] = []

print("\n--- Ridge Regression (varying alpha/lambda) ---")
print(f"{'Alpha':<10} {'MSE':>10} {'MAE':>10} {'R²':>10}")

for alpha in alphas:
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    ridge_pred = ridge.predict(X_test_scaled)
    mse = mean_squared_error(y_test, ridge_pred)
    mae = mean_absolute_error(y_test, ridge_pred)
    r2 = r2_score(y_test, ridge_pred)
    ridge_results.append((alpha, mse, mae, r2))
    print(f"{alpha:<10} {mse:>10.4f} {mae:>10.4f} {r2:>10.4f}")

# Coefficient comparison: Linear vs Ridge at different alphas
feature_names = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

print("\n--- Coefficient Comparison ---")
header = f"{'Feature':<12} {'Linear':>10}"
for alpha in alphas:
    header += f" {'a=' + str(alpha):>10}"
print(header)

ridge_models = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    ridge_models.append(ridge)

for i, name in enumerate(feature_names):
    row = f"{name:<12} {lr.coef_[i]:>10.4f}"
    for model in ridge_models:
        row += f" {model.coef_[i]:>10.4f}"
    print(row)

# Plot: R² vs Alpha
r2_values = [r[3] for r in ridge_results]
plt.figure(figsize=(8, 5))
plt.plot(alphas, r2_values, marker='o', color='steelblue')
plt.axhline(y=r2_score(y_test, lr_pred), color='red', linestyle='--', label='Linear Regression')
plt.xscale('log')
plt.xlabel('Alpha (Lambda)')
plt.ylabel('R² Score')
plt.title('Ridge Regression: R² vs Alpha')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
