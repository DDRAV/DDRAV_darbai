# Užduotis 1: Klasifikuoti, ar navikas yra gerybinis ar piktybinis.
# Sukurkite sprendimo medį naudodami Gini ir entropiją.
# Palyginkite tikslumo rezultatus ir sukurkite klsifikavimo matricas. Vizualizuokite sprendimo medį ir aptarkite svarbiausius kintamuosius.
# Duomenų rinkinys: Naudokite sklearn.datasets.load_breast_cancer().

# Import required libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Duomenų užkrovimas
data = load_breast_cancer()
X = data.data
y = data.target
print(X)
print(y)

# Treniravimo ir testavimo rinkiniai
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sprendimo medis su Gini kriterijumi
clf_gini = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=4)
clf_gini.fit(X_train, y_train)

# Sprendimo medis su Entropijos kriterijumi
clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=4)
clf_entropy.fit(X_train, y_train)

# Random Forest modelis
rf = RandomForestClassifier(n_estimators=3, criterion='gini', bootstrap=True, max_samples=1.0, random_state=42, max_depth=4)
rf.fit(X_train, y_train)

# Prognozės
y_pred_gini = clf_gini.predict(X_test)
y_pred_entropy = clf_entropy.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Tikslumai
accuracy_gini = accuracy_score(y_test, y_pred_gini)
accuracy_entropy = accuracy_score(y_test, y_pred_entropy)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Accuracy (Gini): {accuracy_gini:.4f}")
print(f"Accuracy (Entropy): {accuracy_entropy:.4f}")
print(f"Accuracy (Random Forest): {accuracy_rf:.4f}")

#Confusion matricos
conf_matrix_gini = confusion_matrix(y_test, y_pred_gini)
conf_matrix_entropy = confusion_matrix(y_test, y_pred_entropy)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Spausdiname paprastas confusion matricas
print("\nConfusion Matrix (Gini):")
print(conf_matrix_gini)

print("\nConfusion Matrix (Entropy):")
print(conf_matrix_entropy)

print("\nConfusion Matrix (Random Forest):")
print(conf_matrix_rf)

# Confusion matricų diagramos sudetingesnis
#plt.figure(figsize=(15, 4))

# Gini
#plt.subplot(1, 3, 1)
#plt.imshow(conf_matrix_gini, interpolation='nearest', cmap=plt.cm.Blues)
#plt.title("Confusion Matrix (Gini)")
#plt.colorbar()
#tick_marks = np.arange(len(data.target_names))
#plt.xticks(tick_marks, data.target_names, rotation=45)
#plt.yticks(tick_marks, data.target_names)
#plt.ylabel('True Label')
#plt.xlabel('Predicted Label')

# Entropy
#plt.subplot(1, 3, 2)
#plt.imshow(conf_matrix_entropy, interpolation='nearest', cmap=plt.cm.Blues)
#plt.title("Confusion Matrix (Entropy)")
#plt.colorbar()
#plt.xticks(tick_marks, data.target_names, rotation=45)
#plt.yticks(tick_marks, data.target_names)
#plt.ylabel('True Label')
#plt.xlabel('Predicted Label')

# Random Forest
#plt.subplot(1, 3, 3)
#plt.imshow(conf_matrix_rf, interpolation='nearest', cmap=plt.cm.Blues)
#plt.title("Confusion Matrix (Random Forest)")
#plt.colorbar()
#plt.xticks(tick_marks, data.target_names, rotation=45)
#plt.yticks(tick_marks, data.target_names)
#plt.ylabel('True Label')
#plt.xlabel('Predicted Label')

#plt.tight_layout()
#plt.show()

# Klasifikavimo ataskaitos
print("\nClassification Report (Gini):")
print(classification_report(y_test, y_pred_gini, target_names=data.target_names))

print("\nClassification Report (Entropy):")
print(classification_report(y_test, y_pred_entropy, target_names=data.target_names))

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf, target_names=data.target_names))

# Medžių grafikai
plt.figure(figsize=(20, 10))

# Medžio grafikas Gini modelio
plt.subplot(1, 3, 1)
plot_tree(clf_gini, filled=True, feature_names=data.feature_names, class_names=data.target_names, rounded=True)
plt.title("Decision Tree (Gini)")

# Medžio grafikas Entropy modelio
plt.subplot(1, 3, 2)
plot_tree(clf_entropy, filled=True, feature_names=data.feature_names, class_names=data.target_names, rounded=True)
plt.title("Decision Tree (Entropy)")

# Vieno medžio iš Random Forest grafikas
plt.subplot(1, 3, 3)
plot_tree(rf.estimators_[0], filled=True, feature_names=data.feature_names, class_names=data.target_names, rounded=True)
plt.title("Random Forest Tree (Estimator 1)")

plt.tight_layout()
plt.show()

# Svarbiausi kintamieji visiems modeliams
feature_importance_gini = pd.Series(clf_gini.feature_importances_, index=data.feature_names).sort_values(ascending=False)
feature_importance_entropy = pd.Series(clf_entropy.feature_importances_, index=data.feature_names).sort_values(ascending=False)
feature_importance_rf = pd.Series(rf.feature_importances_, index=data.feature_names).sort_values(ascending=False)

# Grafikai kintamųjų svarbos
plt.figure(figsize=(15, 5))

# Gini
plt.subplot(1, 3, 1)
plt.barh(feature_importance_gini.index, feature_importance_gini.values)
plt.title("Feature Importance (Gini)")
plt.gca().invert_yaxis()

# Entropy
plt.subplot(1, 3, 2)
plt.barh(feature_importance_entropy.index, feature_importance_entropy.values)
plt.title("Feature Importance (Entropy)")
plt.gca().invert_yaxis()

# Random Forest
plt.subplot(1, 3, 3)
plt.barh(feature_importance_rf.index, feature_importance_rf.values)
plt.title("Feature Importance (Random Forest)")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

print("Data type:", type(data))  # Vietoj data.type
print('DESCR: ', data.DESCR)
if hasattr(data, 'frame'):
    print('FRAME:', data.frame)
else:
    print("FRAME attribute not available.")

if hasattr(data, 'data_module'):
    print('MODULE:', data.data_module)
else:
    print("MODULE attribute not available.")