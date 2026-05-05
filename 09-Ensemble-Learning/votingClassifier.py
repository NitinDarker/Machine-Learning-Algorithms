from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()

models = {
    "Logistic": lr,
    "DecisionTree": dt,
    "KNN": knn
}

results = {}

print("--- Individual Models ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    print(f"{name:<15}: {acc:.4f}")


# Ensemble
ensemble = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('dt', dt),
        ('knn', knn)
    ],
    voting='hard'
)

ensemble.fit(X_train, y_train)
ensemble_pred = ensemble.predict(X_test)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

results["Ensemble"] = ensemble_acc

print("\n--- Ensemble ---")
print(f"{'VotingClassifier':<15}: {ensemble_acc:.4f}")

print("\n--- Winner ---")
best_model = max(results, key=results.get)

for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:<15}: {acc:.4f}")

print(f"\nBest Model: {best_model}")