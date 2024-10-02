import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Užkrauname „Diabetes Dataset“
data = load_diabetes()
X = data.data
y = data.target

# Rūšiuojame duomenis pagal priklausomybę
sorted_indices = np.argsort(y)
X_sorted = X[sorted_indices]
y_sorted = y[sorted_indices]

# Padalijame duomenis į treniravimo ir testavimo rinkinius
X_train, X_test, y_train, y_test = train_test_split(X_sorted, y_sorted, test_size=0.2, random_state=42)

# 1. Linijinė regresija be duomenų standartizavimo
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
mse_no_scaling = mean_squared_error(y_test, y_pred)
r2_no_scaling = r2_score(y_test, y_pred)
print(f"Linijinės regresijos MSE be standartizavimo: {mse_no_scaling}")
print(f"Linijinės regresijos R^2 be standartizavimo: {r2_no_scaling}")

# 2. Normalizuojame kintamuosius
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Linijinė regresija su normalizuotais duomenimis
linear_model_scaled = LinearRegression()
linear_model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = linear_model_scaled.predict(X_test_scaled)
mse_with_scaling = mean_squared_error(y_test, y_pred_scaled)
r2_with_scaling = r2_score(y_test, y_pred_scaled)
print(f"Linijinės regresijos MSE su standartizavimu: {mse_with_scaling}")
print(f"Linijinės regresijos R^2 su standartizavimu: {r2_with_scaling}")

