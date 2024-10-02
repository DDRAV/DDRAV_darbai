import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Generate some data
np.random.seed(0)
x = np.random.rand(100, 1) * 10  # Random data for x between 0 and 10
y = 3 - 5 * x + 5 * x**2 + np.random.randn(100, 1)  # Non-linear relationship with noise

# Plot the generated data
plt.scatter(x, y, color='blue')
plt.title("Scatter plot of data")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Apply polynomial transformation
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

print(x_poly)

# Fit the linear model on polynomial features
model = LinearRegression()
model.fit(x_poly, y)

# Sort the data for smooth plotting
x_sorted = np.sort(x, axis=0)
x_poly_sorted = poly.transform(x_sorted)

# Predictions for the sorted data
y_pred_sorted = model.predict(x_poly_sorted)

# Plotting the results
plt.scatter(x, y, color='blue')
plt.plot(x_sorted, y_pred_sorted, color='red', linewidth=2)  # Single red line
plt.title("Polynomial Regression Fit (Degree 2)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Evaluating the model
mse = mean_squared_error(y, model.predict(x_poly))
print(f"Mean Squared Error: {mse}")