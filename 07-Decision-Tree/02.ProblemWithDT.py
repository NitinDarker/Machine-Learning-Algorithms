import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import  plot_tree
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("D:/Code/Machine-Learning-Algorithms/06-Decision-Tree/Social_Network_Ads.csv")

# Drop User ID
df.drop("User ID", axis=1, inplace=True)

# Encode Gender
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

# Features & target
X = df[["Gender","Age", "EstimatedSalary"]]
y = df["Purchased"]

clf = DecisionTreeClassifier() # max_depth=4
clf.fit(X,y)

plot_tree(clf)
plt.show() 