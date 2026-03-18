import numpy as np

cgpa = np.array([
    5.0, 5.2, 5.5, 5.7, 6.0, 6.2, 6.5, 6.7, 6.8, 7.0,
    7.1, 7.3, 7.5, 7.6, 7.8, 8.0, 8.1, 8.3, 8.5, 8.6,
    8.8, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8
])

placement = np.array([
    2.0, 2.2, 2.5, 2.7, 3.0, 3.2, 3.8, 4.0, 4.2, 4.5,
    4.8, 5.2, 5.8, 6.0, 6.5, 7.0, 7.2, 7.8, 8.5, 9.0,
    9.8, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 15.0
])

class GradientDescent:
    def __init__(self, epoch=10000, lr=0.01):
        self.epoch = epoch
        self.learning_rate = lr
        self.m = 0
        self.b = 0
    
    def train(self, X, y):
        n = len(X)
        for i in range(self.epoch):
            y_pred = self.m * X + self.b
            gradient_b = (-2/n) * np.sum(y - y_pred)
            gradient_m = (-2/n) * np.sum(X * (y - y_pred))
            self.b -= self.learning_rate * gradient_b
            self.m -= self.learning_rate * gradient_m
    
    def predict(self, X_test):
        return self.m * X_test + self.b

gd = GradientDescent(epoch=100000, lr=0.001)
gd.train(cgpa,placement)
