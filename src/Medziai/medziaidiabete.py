import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


# Replace the path below with the actual file location
diabetes_data = pd.read_csv("diabetes.csv")

# Features: Age, BMI, Blood Pressure, Insulin
X = diabetes_data[['Age', 'BMI', 'BloodPressure', 'Insulin', 'Pregnancies', 'Glucose', 'SkinThickness', 'DiabetesPedigreeFunction']]
y = diabetes_data['Outcome']  # Assuming 'Outcome' is the target variable indicating diabetes (1: Yes, 0: No)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree with Gini criterion
clf_gini = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=5)
clf_gini.fit(X_train, y_train)

# Decision Tree with Entropy criterion
clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=5)
clf_entropy.fit(X_train, y_train)

# Random Forest model
rf = RandomForestClassifier(n_estimators=50, criterion='gini', bootstrap=True, max_samples=1.0, random_state=42, max_depth=5)
rf.fit(X_train, y_train)

# Predictions
y_pred_gini = clf_gini.predict(X_test)
y_pred_entropy = clf_entropy.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Accuracy
accuracy_gini = accuracy_score(y_test, y_pred_gini)
accuracy_entropy = accuracy_score(y_test, y_pred_entropy)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Accuracy (Gini): {accuracy_gini:.4f}")
print(f"Accuracy (Entropy): {accuracy_entropy:.4f}")
print(f"Accuracy (Random Forest): {accuracy_rf:.4f}")

# Confusion Matrices
conf_matrix_gini = confusion_matrix(y_test, y_pred_gini)
conf_matrix_entropy = confusion_matrix(y_test, y_pred_entropy)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Print simple confusion matrices
print("\nConfusion Matrix (Gini):")
print(conf_matrix_gini)

print("\nConfusion Matrix (Entropy):")
print(conf_matrix_entropy)

print("\nConfusion Matrix (Random Forest):")
print(conf_matrix_rf)

# Classification reports
print("\nClassification Report (Gini):")
print(classification_report(y_test, y_pred_gini))

print("\nClassification Report (Entropy):")
print(classification_report(y_test, y_pred_entropy))

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# Tree visualizations
plt.figure(figsize=(20, 10))

# Gini tree plot
plt.subplot(1, 3, 1)
plot_tree(clf_gini, filled=True, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], rounded=True)
plt.title("Decision Tree (Gini)")

# Entropy tree plot
plt.subplot(1, 3, 2)
plot_tree(clf_entropy, filled=True, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], rounded=True)
plt.title("Decision Tree (Entropy)")

# One tree from Random Forest
plt.subplot(1, 3, 3)
plot_tree(rf.estimators_[0], filled=True, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], rounded=True)
plt.title("Random Forest Tree (Estimator 1)")

plt.tight_layout()
plt.show()

# Feature importance plots
feature_importance_gini = pd.Series(clf_gini.feature_importances_, index=X.columns).sort_values(ascending=False)
feature_importance_entropy = pd.Series(clf_entropy.feature_importances_, index=X.columns).sort_values(ascending=False)
feature_importance_rf = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Feature importance graphs
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

print(type(diabetes_data))  # This will print the type of the object, which should be a pandas DataFrame.

# Print the first few rows of the DataFrame to check the data
print(diabetes_data.head())

# Print the columns of the DataFrame
print(diabetes_data.columns)