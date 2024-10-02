#Užduotis 4 Nuspėkite, ar darbuotojas gaus paaukštinimą, remdamiesi jų darbo patirtimi,
# rezultatų vertinimais ir darbo užduočių sudėtingumu.
# Kintamieji: Darbo patirtis (metai), Rezultatų vertinimas (0-10), Užduočių sudėtingumas (0-10).
# Tikslas (Y): Ar gaus paaukštinimą? (0 = ne, 1 = taip).

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Sukuriame duomenis
data = {
    'Darbo_patirtis': [5, 3, 10, 7, 15, 8, 2, 6, 4, 12, 11, 9, 1, 13, 14, 0],
    'Rezultatu_vertinimas': [8, 6, 9, 7, 10, 6, 5, 8, 4, 9, 7, 6, 3, 8, 10, 4],
    'Uzdaviniu_sudetingumas': [7, 5, 9, 6, 10, 4, 3, 8, 2, 9, 7, 5, 1, 8, 10, 2],
    'Paaukstintas': [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0]
}

df = pd.DataFrame(data)


X = df[['Darbo_patirtis', 'Rezultatu_vertinimas', 'Uzdaviniu_sudetingumas']]
Y = df['Paaukstintas']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Logistinės regresijos modelio sukūrimas ir mokymas
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train, Y_train)

# Prognozės testavimo rinkiniui
y_pred = model.predict(X_test)

# Tikslumo skaičiavimas
accuracy = accuracy_score(Y_test, y_pred)
cm = confusion_matrix(Y_test, y_pred)

# Rezultatai
print("Tikslumas:", accuracy)
print("Klaidų matrica:\n", cm)
print("Prognozės (0 = ne, 1 = taip):", y_pred)
