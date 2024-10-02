#Užduotis 2  Nuspėkite, ar studentas išlaikys egzaminą, remdamiesi jų mokymosi valandomis, vidurkiu
# ir dalyvavimo lygmeniu.
# Kintamieji: Mokymosi valandos, Pažymių vidurkis, Dalyvavimas (%).
# Tikslas (Y): Ar išlaikys? (0 = ne, 1 = taip).

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


data = {
    'Mokymosi_valandos': [10, 5, 12, 8, 15, 4, 6, 11, 7, 9, 14, 3, 2, 13, 8, 10],
    'Pazymiu_vidurkis': [6.5, 4.0, 7.5, 6.0, 8.0, 3.5, 5.5, 7.0, 5.0, 6.5, 8.5, 2.0, 4.5, 7.5, 6.0, 7.0],
    'Dalyvavimas': [80, 60, 90, 70, 95, 50, 65, 85, 75, 80, 92, 40, 30, 88, 70, 80],
    'Islaikes': [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['Mokymosi_valandos', 'Pazymiu_vidurkis', 'Dalyvavimas']]
Y = df['Islaikes']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Logistinės regresijos modelio sukūrimas ir treniravimas
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Prognozių atlikimas
y_pred = model.predict(X_test)  # Numatomos klasės (0 arba 1)
y_proba = model.predict_proba(X_test)[:, 1]  # Tikimybės priskirti klasei 1

# Modelio vertinimas
accuracy = accuracy_score(Y_test, y_pred)
conf_matrix = confusion_matrix(Y_test, y_pred)

# Rezultatų išvedimas
print("Prognozės (0 = ne, 1 = taip):", y_pred)
print("Prognozių tikimybės priskirti klasei '1':", y_proba)
print("Tikslumas:", accuracy)
print("Konfuzijos matrica:\n", conf_matrix)
