from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


digits = load_digits()
X, y = digits.data, digits.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(50, ), activation='relu', max_iter=100, random_state=42)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

print('ACC: ', accuracy_score(y_test, y_pred))
print('CM: ', confusion_matrix(y_test, y_pred))