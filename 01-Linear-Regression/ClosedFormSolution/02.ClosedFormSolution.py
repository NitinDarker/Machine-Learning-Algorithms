import numpy as np
import pandas as pd

# Load cleaned dataset
df = pd.read_csv('cleaned_data.csv')

# ---------------------------
# Simple Linear Regression
# ---------------------------
class SimpleLR:
    def __init__(self):
        self.m = None  # slope
        self.b = None  # intercept
    
    def train(self, X_train, y_train):
        """
        Train the model using closed-form solution
        """
        x = X_train['CGPA']
        y = y_train
        
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        self.m = np.sum((x - x_mean)*(y - y_mean)) / np.sum((x - x_mean)**2)
        self.b = y_mean - self.m * x_mean

    def predict(self, X_test):
        """
        Predict Package for given CGPA
        """
        return self.m * X_test + self.b

# ---------------------------
# Custom Train-Test Split
# ---------------------------
def split_data(X, y, test_size=0.25, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    test_len = int(len(X) * test_size)
    test_idx = indices[:test_len]
    train_idx = indices[test_len:]

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test  = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test  = y.iloc[test_idx].reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test

# ---------------------------
# Prepare Data
# ---------------------------
X = df[['CGPA']]
y = df['Package']

X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=42)

# ---------------------------
# Train Model
# ---------------------------
model = SimpleLR()
model.train(X_train, y_train)

print("Slope (m):", model.m)
print("Intercept (b):", model.b)

# ---------------------------
# Predict
# ---------------------------
y_pred = model.predict(X_test['CGPA'])

# Predict example for CGPA = 8.0
pred_example = model.predict(8.0)
print("Predicted package for CGPA 8.0:", pred_example)

# ---------------------------
# Evaluate Model
# ---------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation Metrics:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# ---------------------------
# Visualize Regression Line
# ---------------------------
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

plt.figure(figsize=(8,6))
plt.scatter(df['CGPA'], df['Package'], color='blue', label='Actual')
plt.plot(df['CGPA'], model.predict(df['CGPA']), color='red', label='Regression Line')
plt.xlabel('CGPA')
plt.ylabel('Package (LPA)')
plt.title('Simple Linear Regression')
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("Model Accuracy (R² Score):", r2)
plt.legend()
plt.show()
