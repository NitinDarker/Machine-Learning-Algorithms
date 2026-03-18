import numpy as np

cgpa = np.array([
    5.0, 5.2, 5.5, 5.7, 6.0, 6.2, 6.5, 6.7, 6.8, 7.0,
    7.1, 7.3, 7.5, 7.6, 7.8, 8.0, 8.1, 8.3, 8.5, 8.6,
    8.8, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8
])  # <-- removed reshape

placement = np.array([
    2.0, 2.2, 2.5, 2.7, 3.0, 3.2, 3.8, 4.0, 4.2, 4.5,
    4.8, 5.2, 5.8, 6.0, 6.5, 7.0, 7.2, 7.8, 8.5, 9.0,
    9.8, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 15.0
])

class FixedSlopeGradientDescent:
    def __init__(self, epoch=100, lr=0.1):
        self.epoch = epoch
        self.learning_rate = lr
        self.m = 2.662
        self.b = -2923
    
    def train(self):
        n = len(cgpa)
        for i in range(self.epoch):
            y_pred = self.m * cgpa + self.b
            gradient_b = (-2/n) * np.sum(placement - y_pred)
            self.b = self.b - self.learning_rate * gradient_b

        print("Final intercept b =", self.b)

myGD = FixedSlopeGradientDescent()
myGD.train()
