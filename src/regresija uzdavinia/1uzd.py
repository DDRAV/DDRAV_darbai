import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Užkrauname „California Housing Dataset“
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Padalijame duomenis į treniravimo ir testavimo rinkinius
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pasirinktas kintamasis, kurio atžvilgiu atvaizduosime grafikus (pvz., 'AveRooms')
selected_feature = 'AveRooms'  #AveRooms/HouseAge

# 1. Pradinis modelis su visais kintamaisiais
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
mse_full = mean_squared_error(y_test, y_pred)
print(f"Pradinio modelio MSE (su visais kintamaisiais): {mse_full}")

# Rūšiuojame duomenis pagal pasirinktą kintamąjį
sorted_idx = np.argsort(X_test[selected_feature])
X_test_sorted = X_test.iloc[sorted_idx]
y_test_sorted = y_test[sorted_idx]
y_pred_sorted = y_pred[sorted_idx]

# Atvaizduojame regresijos liniją pradinei linijinei regresijai
plt.figure(figsize=(8, 6))
plt.scatter(X_test_sorted[selected_feature], y_test_sorted, color='blue', label='Tikros reikšmės')
plt.plot(X_test_sorted[selected_feature], y_pred_sorted, color='red', label='Regresijos linija')
plt.title(f"Regresijos linija prieš pašalinant kintamuosius")
plt.xlabel(selected_feature)
plt.ylabel("MedHouseVal")
plt.legend()
plt.show()

# 2. Pašaliname po vieną kintamąjį ir stebime, kaip keičiasi MSE bei atvaizduojame liniją
for column in X.columns:
    if column == selected_feature:
        continue  # Nenorime pašalinti pasirinkto kintamojo, kurį rodome grafike

    X_train_reduced = X_train.drop(columns=[column])
    X_test_reduced = X_test.drop(columns=[column])

    linear_model_reduced = LinearRegression()
    linear_model_reduced.fit(X_train_reduced, y_train)
    y_pred_reduced = linear_model_reduced.predict(X_test_reduced)

    mse_reduced = mean_squared_error(y_test, y_pred_reduced)
    print(f"Pašalinus kintamąjį {column}, MSE: {mse_reduced}")

    # Rūšiuojame duomenis pagal pasirinktą kintamąjį
    X_test_reduced_sorted = X_test_reduced.iloc[sorted_idx]  # Rūšiuojame pagal pradžioje gautą indeksą
    y_pred_reduced_sorted = y_pred_reduced[sorted_idx]

    # Atvaizduojame regresijos liniją po kintamojo pašalinimo
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test_sorted[selected_feature], y_test_sorted, color='blue', label='Tikros reikšmės')
    plt.plot(X_test_sorted[selected_feature], y_pred_reduced_sorted, color='red', label='Regresijos linija')
    plt.title(f"Regresijos linija po pašalinimo {column}")
    plt.xlabel(selected_feature)
    plt.ylabel("MedHouseVal")
    plt.legend()
    plt.show()
