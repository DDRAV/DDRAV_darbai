#Sukurkite modelį, kuris nuspės, ar klientas pirks produktą,
# remdamasis tokiomis savybėmis kaip amžius, pajamos, lytis, ir internetinių parduotuvių naršymo laikas.
#Kintamieji: Amžius, Pajamos, Lytis (0 = moteris, 1 = vyras), Naršymo laikas (minutės).
#Tikslas (Y): Nustatyti ar klientas pirks produktą? (0 = ne, 1 = taip).

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    'Amzius': [25, 40, 35, 50, 23, 45, 30, 31, 34, 40, 28, 37, 24, 29, 55, 42],
    'Pajamos': [30000, 50000, 45000, 60000, 28000, 58000, 35000, 36000, 40000, 52000, 33000, 49000, 31000, 42000, 70000,
                48000],
    'Lytis': [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0],
    'Narsymo_laikas': [30, 80, 55, 120, 20, 90, 45, 50, 40, 100, 35, 70, 25, 60, 130, 75],
    'Pirko': [0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

X = df[['Amzius', 'Pajamos', 'Lytis', 'Narsymo_laikas']]
Y = df['Pirko']

# Sukuriame du duomenu rinkinius
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

# Implementuojame modeli
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train, Y_train)

threshold = 0.0095


# Prognozuojame
y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)[:, 1]

y_cursom_th = []
for i in y_proba:
    if i >= threshold:
        y_cursom_th.append(1)
    else:
        y_cursom_th.append(0)


print(y_pred, y_proba)

acc = accuracy_score(y_test, y_cursom_th)
cm = confusion_matrix(y_test, y_cursom_th)

print(acc)
print(cm)

