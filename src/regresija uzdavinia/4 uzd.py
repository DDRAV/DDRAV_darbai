import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Užkrauname „California Housing Dataset“
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Rūšiuojame duomenis pagal target reikšmes
sorted_indices = np.argsort(y)
X_sorted = X.iloc[sorted_indices]
y_sorted = y[sorted_indices]

# Padalijame duomenis į treniravimo ir testavimo rinkinius
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Išsaugosime MSE rezultatus palyginimui
mse_results = []

# Modelio pritaikymas skirtingais laipsniais (nuo 1 iki 5)
for degree in range(1, 6):
    # Polinominės funkcijos transformacija
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Linijinė regresija polinominėms funkcijoms
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    # Numatymas ir MSE skaičiavimas
    y_pred_poly = poly_model.predict(X_test_poly)
    mse_poly = mean_squared_error(y_test, y_pred_poly)

    # Išsaugome MSE rezultatą
    mse_results.append((degree, mse_poly))
    print(f"Polinominės regresijos MSE (laipsnis {degree}): {mse_poly}")

# Vizualizuojame rezultatus
degrees = [result[0] for result in mse_results]
mse_values = [result[1] for result in mse_results]

plt.plot(degrees, mse_values, marker='o', linestyle='--', color='b')
plt.title("Polinominės regresijos laipsnio įtaka MSE")
plt.xlabel("Polinominės regresijos laipsnis")
plt.ylabel("MSE")
plt.grid(True)
plt.show()