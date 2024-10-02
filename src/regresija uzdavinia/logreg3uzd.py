#Užduotis 3 Nuspėkite, ar darbuotojas liks įmonėje, remdamiesi amžiumi, darbo patirtimi, atlyginimu ir pasitenkinimo darbu lygiu.
#Kintamieji: Amžius, Darbo patirtis (metai), Atlyginimas, Pasitenkinimas darbu (0-10).
#Tikslas (Y): Ar liks įmonėje? (0 = ne, 1 = taip).

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Sukuriame duomenis
data = {
    'Amzius': [25, 40, 35, 50, 23, 45, 30, 31, 34, 40, 28, 37, 24, 29, 55, 42],
    'Darbo_patirtis': [2, 10, 7, 15, 1, 12, 5, 3, 6, 10, 3, 8, 1, 4, 18, 9],
    'Atlyginimas': [30000, 50000, 45000, 60000, 28000, 58000, 35000, 36000, 40000, 52000, 33000, 49000, 31000, 42000, 70000, 48000],
    'Pasitenkinimas_darbu': [8, 6, 7, 5, 9, 4, 7, 6, 6, 5, 8, 6, 9, 7, 5, 6],
    'Liks_imoneje': [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['Amzius', 'Darbo_patirtis', 'Atlyginimas', 'Pasitenkinimas_darbu']]
Y = df['Liks_imoneje']

# Duomenų padalijimas
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