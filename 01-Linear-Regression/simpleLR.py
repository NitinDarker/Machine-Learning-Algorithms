import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('D:/Code/ML/Machine-Learning-Algorithms/01-Linear-Regression/placement.csv')
# print(df.head())

plt.scatter(df['cgpa'],df['package'])
plt.xlabel("CGPA")
plt.ylabel("Placement in LPAs")
# plt.show()

x = df.iloc[:,0:1]
y = df.iloc[:,-1]

# print(x)
# print(y)

from sklearn.model_selection  import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=2)
# print(X_train)
print(X_test.shape)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,Y_train) #training of the model

# print(X_test)

predication = lr.predict(X_test.iloc[0].values.reshape(1,1))

print("Predicated placement:",predication)

plt.scatter(df['cgpa'],df['package'])
plt.plot(X_train,lr.predict(X_train))
plt.show()

#slope of the line
print(lr.coef_)
#print the y intersect of the line

print(lr.intercept_)