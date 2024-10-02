#Užduotis 1:
#Sugeneruokite duomenų rinkinį, kuriame santykis tarp
#𝑥 ir  𝑦 atitinka kubinę priklausomybę (pvz. y=ax3+bx2+cx+d su triukšmu). Pritaikykite 3 laipsnio polinominės regresijos modelį šiems duomenims ir vizualizuokite rezultatą.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Sugeneruojami kubiniai duomenys su triukšmu
np.random.seed(0)
x = np.random.rand(100, 1) * 10  # Atsitiktiniai x duomenys nuo 0 iki 10
y = 2 - 3 * x + 4 * x**2 - 0.5 * x**3 + np.random.randn(100, 1)  # Kubinė priklausomybė su triukšmu

# Duomenų grafikas
plt.scatter(x, y, color='blue')
plt.title("3 laipsnis")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# 3 laipsnio polinominė transformacija
poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x)

# Tiesinės regresijos modelio treniravimas naudojant polinomines ypatybes
model = LinearRegression()
model.fit(x_poly, y)

# Surūšiuoti duomenys sklandžiam grafiko atvaizdavimui
x_sorted = np.sort(x, axis=0)
x_poly_sorted = poly.transform(x_sorted)

# Prognozės remiantis modeliu
y_pred_sorted = model.predict(x_poly_sorted)

# Rezultatų grafikas
plt.scatter(x, y, color='blue')
plt.plot(x_sorted, y_pred_sorted, color='red', linewidth=2)  # Raudona linija rodo modelio prognozę
plt.title("Polinominė regresija (3 laipsnio)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Modelio vertinimas
mse = mean_squared_error(y, model.predict(x_poly))
print(f"Vidutinė kvadratinė klaida: {mse}")


#Užduotis 2:
#Pritaikyti ant 1 užd. sukurtų duomenų tiesinę regresiją. Apskaičiuokite MSE koeficientus ir palyginkite rezultatus.
linear_model = LinearRegression()
linear_model.fit(x, y)

# Prognozės naudojant tiesinę regresiją
y_pred_linear = linear_model.predict(x)

# Rezultatų grafikas su tiesine regresija
plt.scatter(x, y, color='blue')
plt.plot(x_sorted, linear_model.predict(x_sorted), color='green', linewidth=2)  # Žalia linija rodo tiesinę regresiją
plt.title("Tiesinės regresijos rezultatas")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# MSE apskaičiavimas tiesinei regresijai
mse_linear = mean_squared_error(y, y_pred_linear)
print(f"Tiesinės regresijos vidutinė kvadratinė klaida: {mse_linear}")

# Palyginimas su 3 laipsnio polinominės regresijos MSE
print(f"Polinominės regresijos (3 laipsnio) vidutinė kvadratinė klaida: {mse}")


#Užduotis 3:
#Naudodami tikrus duomenis (pvz., iš sklearn bibliotekos, tokius kaip „California housing“ ar „diabetes dataset“),
# nustatykite, ar polinominė regresija geriau pritaiko duomenis nei linijinė regresija. Eksperimentuokite su polinominės regresijos laipsniais 2, 3 ir 4.

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Užkrauname „Diabetes“ duomenų rinkinį
data = load_diabetes()
X = data.data[:, 2:3]  # Naudosime vieną kintamąjį (pvz., BMI)
y = data.target

# Padalijame duomenis į treniravimo ir testavimo rinkinius
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 1. Tiesinė regresija
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)

# 2. Polinominė regresija (laipsnis 2)
poly2 = PolynomialFeatures(degree=2)
X_poly2_train = poly2.fit_transform(X_train)
X_poly2_test = poly2.transform(X_test)

model_poly2 = LinearRegression()
model_poly2.fit(X_poly2_train, y_train)
y_pred_poly2 = model_poly2.predict(X_poly2_test)
mse_poly2 = mean_squared_error(y_test, y_pred_poly2)

# 3. Polinominė regresija (laipsnis 3)
poly3 = PolynomialFeatures(degree=3)
X_poly3_train = poly3.fit_transform(X_train)
X_poly3_test = poly3.transform(X_test)

model_poly3 = LinearRegression()
model_poly3.fit(X_poly3_train, y_train)
y_pred_poly3 = model_poly3.predict(X_poly3_test)
mse_poly3 = mean_squared_error(y_test, y_pred_poly3)

# 4. Polinominė regresija (laipsnis 4)
poly4 = PolynomialFeatures(degree=4)
X_poly4_train = poly4.fit_transform(X_train)
X_poly4_test = poly4.transform(X_test)

model_poly4 = LinearRegression()
model_poly4.fit(X_poly4_train, y_train)
y_pred_poly4 = model_poly4.predict(X_poly4_test)
mse_poly4 = mean_squared_error(y_test, y_pred_poly4)

# Spausdiname rezultatus
print(f"Tiesinės regresijos MSE: {mse_linear}")
print(f"Polinominės regresijos MSE (laipsnis 2): {mse_poly2}")
print(f"Polinominės regresijos MSE (laipsnis 3): {mse_poly3}")
print(f"Polinominės regresijos MSE (laipsnis 4): {mse_poly4}")

# Grafikai
plt.scatter(X_test, y_test, color='blue', label="Tikri duomenys")
plt.plot(X_test, y_pred_linear, color='green', label="Tiesinė regresija")
plt.plot(X_test, y_pred_poly2, color='red', label="Polinominė regresija (2 laipsnis)")
plt.plot(X_test, y_pred_poly3, color='orange', label="Polinominė regresija (3 laipsnis)")
plt.plot(X_test, y_pred_poly4, color='purple', label="Polinominė regresija (4 laipsnis)")
plt.title("Regresijos modelių palyginimas")
plt.xlabel("BMI")
plt.ylabel("Diabeto progresavimas")
plt.legend()
plt.show()