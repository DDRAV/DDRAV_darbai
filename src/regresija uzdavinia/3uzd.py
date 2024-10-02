import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Užkrauname „California Housing Dataset“
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Rūšiuojame duomenis pagal target reikšmes
sorted_indices = np.argsort(y)
X_sorted = X.iloc[sorted_indices]
y_sorted = y[sorted_indices]

# Padalijame duomenis į treniravimo ir testavimo rinkinius
X_train, X_test, y_train, y_test = train_test_split(X_sorted, y_sorted, test_size=0.2, random_state=42)

# 1. Paprasta linijinė regresija
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
print(f"Linijinės regresijos MSE: {mse_linear}")

# 2. Polinominė regresija su laipsniu 2
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
print(f"Polinominės regresijos MSE (laipsnis 2): {mse_poly}")

# Palyginimas
if mse_poly < mse_linear:
    print("Polinominė regresija nepagerina modelio tikslumą.")
else:
    print("Polinominė regresija pagerina modelio tikslumo.")
