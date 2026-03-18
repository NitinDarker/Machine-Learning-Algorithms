class MyLR:
    def __init__(self):
        self.m=None
        self.b =None

    def fit(self,X_train,Y_train):
        num=0
        den=0
        
        for i in range(X_train.shape[0]):
            num = num + ((X_train[i] - X_train.mean())*(Y_train[i] - Y_train.mean()))
            den = den + (X_train[i]-X_train.mean())**2
        
        self.m = num / den
        self.b = Y_train.mean() - self.m*(X_train.mean())

        print("Slope is:",self.m)
        print("Y intersect is:",self.b)
    
    def predict(self,X_test):
        print("Predict value is:",self.m*X_test + self.b)
       
import numpy as np
import pandas as pd

df = pd.read_csv('D:/Code/ML/Machine-Learning-Algorithms/01-Linear-Regression/placement.csv')
# print(df)

X = df.iloc[:,0].values
Y = df.iloc[:,1].values
# print(X)
# print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)
# print(X_train)

# print(Y_train)
lr = MyLR()
lr.fit(X_train,Y_train)
lr.predict(9.0)