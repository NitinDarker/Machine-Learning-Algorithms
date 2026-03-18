from sklearn.datasets import load_iris
iris = load_iris()

 
# print(iris.data) # input data X
X = iris.data 
y = iris.target

# print(y)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

print(X_train.shape)
print(X_test.shape)


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred))

from sklearn.tree import  plot_tree
import matplotlib.pyplot as plt


plot_tree(clf)
plt.show()
