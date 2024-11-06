#import kagglehub
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


# Download latest version
#path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
#print("Path to dataset files:", path)

# Užkrauti „Credit Card Fraud“ duomenų rinkinį
data = pd.read_csv('C:/Users/drawn/Mokymai/DDRAV Mokymai/DDRAV Darbai/PY_13_paskaita_darius/src/Balansavimas/creditcard.csv')  # Pakeiskite su tinkamu duomenų failo keliu
X = data.drop("Class", axis=1)
y = data["Class"]

# 1. Nesubalansuotų duomenų modelis
print("1. Nesubalansuotų duomenų modelis")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 2. Subalansuotas su Random Oversampling
print("\n2. Subalansuotas su Random Oversampling")
oversampler = RandomOverSampler()
X_res, y_res = oversampler.fit_resample(X, y)
print("Balanced classes:", Counter(y_res))
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 3. Subalansuotas su Random Undersampling
print("\n3. Subalansuotas su Random Undersampling")
undersampler = RandomUnderSampler()
X_res, y_res = undersampler.fit_resample(X, y)
print("Balanced classes:", Counter(y_res))
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 4. Subalansuotas su SMOTE
print("\n4. Subalansuotas su SMOTE")
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)
print("Balanced classes:", Counter(y_res))
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 5. Subalansuotas su Class Weight Adjustment
print("\n5. Subalansuotas su Class Weight Adjustment")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model_weighted = RandomForestClassifier(class_weight='balanced', random_state=42)
model_weighted.fit(X_train, y_train)
y_pred = model_weighted.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))