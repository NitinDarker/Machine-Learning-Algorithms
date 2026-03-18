import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("D:/Code/ML/Machine-Learning-Algorithms/06-Decision-Tree/Social_Network_Ads.csv")

# Drop User ID
df.drop("User ID", axis=1, inplace=True)

# Encode Gender
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

# Features & target
X = df[["Gender","Age", "EstimatedSalary"]]
y = df["Purchased"]


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier() # max_depth=4

clf.fit(X,y)



from sklearn.tree import  plot_tree
import matplotlib.pyplot as plt


plot_tree(clf)
plt.show() # overfitting


