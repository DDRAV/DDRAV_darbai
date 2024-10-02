#1. Nuspėkite, ar keleivis išgyveno Titaniko katastrofą (Titaniko duomenų rinkinys)
#Kintamieji: Keleivio amžius, lytis, klasė, bilieto kaina.
#Tikslas: Nuspėti, ar keleivis išgyveno.
#Užd: Sukurkite sprendimo medį naudodami Gini indeksą ir entropiją. Palyginkite abiejų modelių tikslumą. Vizualizuokite sprendimo medžius.
#Duomenų rinkinys: Naudokite "titanic.csv" (populiarus duomenų rinkinys iš Kaggle).


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

titanic_df = pd.read_csv('titanic.csv')

# duomenu paruosimas
# iterpiam vietoi amziaus medianini amziu
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median())

# Konvertuojam lyti vyras- 0, moteris -1
label_encoder = LabelEncoder()
titanic_df['Sex'] = label_encoder.fit_transform(titanic_df['Sex'])

features = ['Pclass', 'Sex', 'Age', 'Fare']
X = titanic_df[features]
y = titanic_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Medziai
# Treniravimas pgl Gini
clf_gini3 = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=3)
clf_gini3.fit(X_train, y_train)

clf_gini5 = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=5)
clf_gini5.fit(X_train, y_train)

# Treniravimas pgl Entropija
clf_entropy3 = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=3)
clf_entropy3.fit(X_train, y_train)

clf_entropy5 = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=5)
clf_entropy5.fit(X_train, y_train)

# Prognozes ir tikslumas
y_pred_gini3 = clf_gini3.predict(X_test)
y_pred_gini5 = clf_gini5.predict(X_test)

y_pred_entropy3 = clf_entropy3.predict(X_test)
y_pred_entropy5 = clf_entropy5.predict(X_test)

accuracy_gini3 = accuracy_score(y_test, y_pred_gini3)
accuracy_gini5 = accuracy_score(y_test, y_pred_gini5)

accuracy_entropy3 = accuracy_score(y_test, y_pred_entropy3)
accuracy_entropy5 = accuracy_score(y_test, y_pred_entropy5)

print(f'Accuracy using Gini index depth 3: {accuracy_gini3 * 100:.2f}%')
print(f'Accuracy using Gini index depth 5: {accuracy_gini5 * 100:.2f}%')
print(f'Accuracy using Entropy depth 3: {accuracy_entropy3 * 100:.2f}%')
print(f'Accuracy using Entropy depth 5: {accuracy_entropy5 * 100:.2f}%')

# Vizualizacija
# Depth 3
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plot_tree(clf_gini3, feature_names=features, filled=True, class_names=['Not Survived', 'Survived'])
plt.title('Decision Tree (Gini Index) - Max Depth 3')

plt.subplot(1, 2, 2)
plot_tree(clf_entropy3, feature_names=features, filled=True, class_names=['Not Survived', 'Survived'])
plt.title('Decision Tree (Entropy) - Max Depth 3')

plt.show()

# Depth 5
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plot_tree(clf_gini5, feature_names=features, filled=True, class_names=['Not Survived', 'Survived'])
plt.title('Decision Tree (Gini Index) - Max Depth 5')

plt.subplot(1, 2, 2)
plot_tree(clf_entropy5, feature_names=features, filled=True, class_names=['Not Survived', 'Survived'])
plt.title('Decision Tree (Entropy) - Max Depth 5')

plt.show()