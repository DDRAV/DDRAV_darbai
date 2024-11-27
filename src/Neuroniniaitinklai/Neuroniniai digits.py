import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Duomenų paruošimas
digits = load_digits()
X, y = digits.data, digits.target
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# Funkcija eksperimentams su parametrais
def evaluate_params(hidden_layer_sizes_list, activation_list, max_iter_list):
    results = []

    for hidden_layer_sizes in hidden_layer_sizes_list:
        for activation in activation_list:
            for max_iter in max_iter_list:
                mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                                    max_iter=max_iter, random_state=42)
                mlp.fit(X_train, y_train)
                y_pred = mlp.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                results.append((hidden_layer_sizes, activation, max_iter, acc))
    return results


# Parametrų sąrašai
hidden_layer_sizes_list = [(30,), (50,), (100,), (30, 30), (70, 30), (30, 70), (70, 70), (200,), (30, 50, 70), (30, 70, 50), (70, 50, 30), (70, 30, 50)]
activation_list = ['relu', 'logistic', 'tanh']
max_iter_list = [500, 1000, 2000]

# Eksperimentai
results = evaluate_params(hidden_layer_sizes_list, activation_list, max_iter_list)

# Rezultatų atvaizdavimas
for activation in activation_list:
    accuracies = []
    param_combinations = []

    for hidden_layer_sizes, act, max_iter, acc in results:
        if act == activation:
            accuracies.append(acc)
            param_combinations.append(f"{hidden_layer_sizes}-{max_iter}")

    plt.plot(param_combinations, accuracies, label=f"Activation: {activation}")
    plt.xticks(rotation=90)

plt.title("Tikslumo priklausomybė nuo parametrų")
plt.xlabel("Parametrų kombinacijos")
plt.ylabel("Tikslumas")
plt.legend()
plt.tight_layout()
plt.show()
