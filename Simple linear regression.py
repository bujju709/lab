import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 

def simple_linear_regression(X, y): 
    n = len(X) 
    mean_x, mean_y = np.mean(X), np.mean(y) 
# Calculating slope (m) and intercept (b) 
    numerator = np.sum((X - mean_x) * (y - mean_y)) 
    denominator = np.sum((X - mean_x) ** 2) 
    m = numerator / denominator 
    b = mean_y - m * mean_x 
    return m, b 

def predict(X, m, b): 
    return m * X + b 
# Generate synthetic dataset 
np.random.seed(42) 
X = np.random.rand(100) * 10  # Feature variable 
y = 3 * X + 7 + np.random.randn(100) * 2  # Target variable with noise 
# Splitting dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# Train the model 
m, b = simple_linear_regression(X_train, y_train) 
# Make predictions 
y_pred = predict(X_test, m, b) 
# Evaluate the model 
mse = mean_squared_error(y_test, y_pred) 
print(f"Slope (m): {m:.2f}") 
print(f"Intercept (b): {b:.2f}") 
print(f"Mean Squared Error: {mse:.2f}") 
# Visualization 
plt.scatter(X_test, y_test, color='blue', label='Actual') 
plt.plot(X_test, y_pred, color='red', label='Predicted Line') 
plt.xlabel('Feature') 
plt.ylabel('Target') 
plt.title('Simple Linear Regression') 
plt.legend() 
plt.show()