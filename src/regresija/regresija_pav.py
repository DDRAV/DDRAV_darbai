import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = np.array([2, 3, 5, 7, 8]).reshape(-1, 1)

Y = np.array([50, 55, 70, 80, 85])

model = LinearRegression()

model.fit(X, Y)

beta_1 = model.coef_[0]

beta_0 = model.intercept_
print(beta_1,beta_0)


Y_pred = model.predict(X)

r_squared = model.score(X, Y)

print(f"R-squared: {r_squared:.4f}")

plt.scatter(X, Y, color='blue', label='Data points')

plt.plot(X, Y_pred, color='red', linewidth=2, label=f'Regression line: Y = {beta_0:.2f} + {beta_1:.2f}X')

plt.xlabel('Hours Studied')

plt.ylabel('Scores')

plt.title('Simple Linear Regression')

plt.legend()

plt.grid(True)

plt.show()