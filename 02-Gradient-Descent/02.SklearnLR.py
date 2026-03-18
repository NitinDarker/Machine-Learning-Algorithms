import numpy as np
from sklearn.linear_model import LinearRegression

cgpa = np.array([
    5.0, 5.2, 5.5, 5.7, 6.0, 6.2, 6.5, 6.7, 6.8, 7.0,
    7.1, 7.3, 7.5, 7.6, 7.8, 8.0, 8.1, 8.3, 8.5, 8.6,
    8.8, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8
]).reshape(-1, 1)

placement = np.array([
    2.0, 2.2, 2.5, 2.7, 3.0, 3.2, 3.8, 4.0, 4.2, 4.5,
    4.8, 5.2, 5.8, 6.0, 6.5, 7.0, 7.2, 7.8, 8.5, 9.0,
    9.8, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 15.0
])

model = LinearRegression()
model.fit(cgpa, placement)

print(f"Slope (m):     {model.coef_[0]:.4f}")
print(f"Intercept (c): {model.intercept_:.4f}")
print(f"Prediction for CGPA 9.0: {model.predict([[9.0]])[0]:.2f}")
