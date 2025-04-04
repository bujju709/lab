import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.metrics import mean_squared_error 

np.random.seed(42) 
X = np.sort(5 * np.random.rand(20, 1), axis=0)  
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

k = 3 
knn_regressor = KNeighborsRegressor(n_neighbors=k, weights='uniform') 
knn_regressor.fit(X_train, y_train) 

y_pred = knn_regressor.predict(X_test) 

mse = mean_squared_error(y_test, y_pred) 
print(f"Mean Squared Error: {mse:.4f}") 

plt.scatter(X_train, y_train, color='blue', label='Training Data') 
plt.scatter(X_test, y_test, color='green', label='Test Data') 
plt.scatter(X_test, y_pred, color='red', label='Predictions') 
plt.xlabel("Feature") 
plt.ylabel("Target") 
plt.title(f"KNN Regression (k={k})") 
plt.legend() 
plt.show()
