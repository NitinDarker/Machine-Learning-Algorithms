import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def analyzer(max_depth):
    # Load data
    df = pd.read_csv("Social_Network_Ads.csv")

    # Use ONLY two features for visualization
    X = df[["Age", "EstimatedSalary"]].values
    y = df["Purchased"].values

    # Train model
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    clf.fit(X, y)

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1000, X[:, 1].max() + 1000

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.5),
        np.arange(y_min, y_max, 100)
    )

    # Predict over grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=plt.cm.coolwarm)
    plt.xlabel("Age")
    plt.ylabel("Estimated Salary")
    plt.title(f"Decision Tree (max_depth = {max_depth})")
    plt.show()

# Try different depths
analyzer(1)   # Underfitting
analyzer(3)   # Good fit
analyzer(10)  # Overfitting
