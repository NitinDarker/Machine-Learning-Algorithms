import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)

n_samples = 120
n_features = 50

X = np.random.randn(n_samples, n_features)

true_weights = np.zeros(n_features)
true_weights[:5] = [5, -3, 2, 0, 4]

y = X @ true_weights + np.random.normal(0, 10, size=n_samples)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("--- Linear Regression ---")
print(f"{'MSE':<10}: {mean_squared_error(y_test, lr_pred):.4f}")
print(f"{'MAE':<10}: {mean_absolute_error(y_test, lr_pred):.4f}")
print(f"{'R²':<10}: {r2_score(y_test, lr_pred):.4f}")


alphas = [0.01, 0.1, 1, 10, 50, 100]
lasso_results = []

print("\n--- Lasso Regression (varying alpha) ---")
print(f"{'Alpha':<10} {'MSE':>10} {'MAE':>10} {'R²':>10}")

lasso_models = {}

for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    lasso_results.append((alpha, mse, mae, r2))
    lasso_models[alpha] = model

    print(f"{alpha:<10} {mse:>10.4f} {mae:>10.4f} {r2:>10.4f}")


print("\n--- Coefficient Comparison (first 10 features) ---")

header = f"{'Feature':<10} {'Linear':>10}"
for alpha in alphas:
    header += f" {'a=' + str(alpha):>10}"
print(header)

for i in range(10):
    row = f"f{i:<9} {lr.coef_[i]:>10.4f}"
    for alpha in alphas:
        row += f" {lasso_models[alpha].coef_[i]:>10.4f}"
    print(row)


print("\n--- Sparsity (Zero Coefficients) ---")
for alpha in alphas:
    zeros = np.sum(lasso_models[alpha].coef_ == 0)
    print(f"alpha={alpha:<6} -> {zeros} / {n_features} coefficients are zero")


best_alpha = min(lasso_results, key=lambda x: x[1])[0]
print(f"\nBest alpha based on MSE: {best_alpha}")