import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
X, y = load_breast_cancer(return_X_y=True)
X = np.array(X)
y = np.array(y)
target_names = ["malignant", "benign"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)
knn_acc = accuracy_score(y_test, knn_pred)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

# Results
print("--- KNN (k=5) ---")
print(f"Accuracy: {knn_acc:.4f}")
print(classification_report(y_test, knn_pred, target_names=target_names, digits=4))

print("--- Decision Tree ---")
print(f"Accuracy: {dt_acc:.4f}")
print(classification_report(y_test, dt_pred, target_names=target_names, digits=4))

# Show exact prediction differences
diff_count = int(np.sum(knn_pred != dt_pred))
print(f"Number of predictions where KNN and DT disagree: {diff_count} out of {len(y_test)}")

# Bar chart comparison
plt.figure(figsize=(6, 4))
plt.bar(['KNN (k=5)', 'Decision Tree'], [float(knn_acc), float(dt_acc)], color=['steelblue', 'coral'])
plt.ylabel('Accuracy')
plt.title('KNN vs Decision Tree')
plt.ylim(0.85, 1.0)
plt.tight_layout()
plt.show()
