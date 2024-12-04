from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define the parameter grid for optimization
param_grid = {
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'learning_rate_init': [0.001, 0.005, 0.01, 0.02],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'momentum': [0.5, 0.9, 0.99],
    'activation': ['relu'],  # Pridedame RELU kaip aktyvacijos funkciją
    'alpha': [0.0001, 0.001, 0.01]  # Pridedame reguliavimo stiprumo reikšmes
}

# Initialize the MLPClassifier
mlp = MLPClassifier(max_iter=2000, random_state=42)

# Set up GridSearchCV to find the best parameters
grid_search = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=2,
    n_jobs=-1,
    return_train_score=True  # Ensure training scores are computed
)

# Fit the model using GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and evaluate on the test set
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
test_predictions = best_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, test_predictions))

# Extract results from GridSearchCV
results = pd.DataFrame(grid_search.cv_results_)

#1 Pivot data for heatmap
heatmap_data = results.pivot_table(
    values='mean_test_score',
    index='param_learning_rate_init',
    columns='param_solver'
)

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".3f")
plt.title("Accuracy Heatmap")
plt.xlabel("Solver")
plt.ylabel("Learning Rate Init")
plt.show()

#2 Plot confusion matrix
# Generate predictions using cross-validation
y_pred_cv = cross_val_predict(best_model, X, y, cv=5)

# Compute and print the confusion matrix
conf_matrix = confusion_matrix(y, y_pred_cv)
print("Cross-Validation Confusion Matrix:")
print(conf_matrix)

# (Optional) Visualize the confusion matrix
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=data.target_names).plot(cmap="Blues")
plt.title("Cross-Validation Confusion Matrix")
plt.show()

# Plot learning_rate_init vs accuracy
plt.figure(figsize=(10, 6))
for solver in param_grid['solver']:
    subset = results[results['param_solver'] == solver]
    plt.plot(
        subset['param_learning_rate_init'],
        subset['mean_test_score'],
        marker='o',
        label=f"Solver: {solver}"
    )

plt.title("Learning Rate Init vs Accuracy")
plt.xlabel("Learning Rate Init")
plt.ylabel("Mean CV Accuracy")
plt.legend()
plt.grid()
plt.show()

# Plot solver vs accuracy
plt.figure(figsize=(10, 6))
mean_scores = results.groupby('param_solver')['mean_test_score'].mean()
plt.bar(mean_scores.index, mean_scores.values, color='skyblue')
plt.title("Solver Comparison")
plt.xlabel("Solver")
plt.ylabel("Mean CV Accuracy")
plt.grid(axis='y')
plt.show()

#3 Tikslumo dėžių diagrama (Boxplot) pagal solver
plt.figure(figsize=(10, 6))
sns.boxplot(x='param_solver', y='mean_test_score', data=results, palette="Set2", legend=False)
plt.title("Solver Accuracy Distribution")
plt.xlabel("Solver")
plt.ylabel("Accuracy")
plt.show()


#4 Klaidų skirtumas tarp mokymo ir testavimo rinkinių
train_score = grid_search.cv_results_['mean_train_score']
test_score = grid_search.cv_results_['mean_test_score']

plt.figure(figsize=(10, 6))
plt.plot(train_score, label='Training Accuracy', marker='o')
plt.plot(test_score, label='Test Accuracy', marker='x')
plt.title("Training vs Test Accuracy")
plt.xlabel("Parameter Combination Index")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()


