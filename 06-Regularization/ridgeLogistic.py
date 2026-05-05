import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MyLogisticRidge:
    def __init__(self, lr=0.01, epochs=1000, alpha=1.0):
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha
        self.weights = None
        self.bias = 0.0

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.epochs):
            z = X @ self.weights + self.bias
            preds = self.sigmoid(z)

            # gradients
            dw = (1/n_samples) * (X.T @ (preds - y))
            db = (1/n_samples) * np.sum(preds - y)

            # L2 regularization
            dw += (self.alpha / n_samples) * self.weights

            # update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        return self.sigmoid(X @ self.weights + self.bias)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


np.random.seed(42)

n_samples = 200
n_features = 50

X = np.random.randn(n_samples, n_features)

true_w = np.zeros(n_features)
true_w[:5] = [3, -2, 1, 0, 2]

logits = X @ true_w
probs = 1 / (1 + np.exp(-logits))

y = (probs > 0.5).astype(int)

# Add noise
flip_idx = np.random.choice(n_samples, size=40, replace=False)
y[flip_idx] = 1 - y[flip_idx]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


sk_model = LogisticRegression(max_iter=1000)
sk_model.fit(X_train, y_train)
sk_pred = sk_model.predict(X_test)

print("--- Sklearn Logistic ---")
print("Accuracy:", accuracy_score(y_test, sk_pred))
print("F1:", f1_score(y_test, sk_pred))


alphas = [0.01, 0.1, 1, 10, 50]

print("\n--- Custom Ridge Logistic ---")
print(f"{'Alpha':<10} {'Acc':>10} {'F1':>10}")

models = {}

for alpha in alphas:
    model = MyLogisticRidge(alpha=alpha, lr=0.05, epochs=2000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    models[alpha] = model
    print(f"{alpha:<10} {acc:>10.4f} {f1:>10.4f}")


print("\n--- Coefficient Comparison (first 10 features) ---")

header = f"{'Feature':<10} {'Sklearn':>10}"
for alpha in alphas:
    header += f" {'a=' + str(alpha):>10}"
print(header)

for i in range(10):
    row = f"f{i:<9} {sk_model.coef_[0][i]:>10.4f}"
    for alpha in alphas:
        row += f" {models[alpha].weights[i]:>10.4f}"
    print(row)