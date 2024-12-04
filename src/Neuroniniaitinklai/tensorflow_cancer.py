import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam
from tensorflow.keras.losses import Huber, MeanSquaredError
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 1. Duomenų įkėlimas
data = load_breast_cancer()
X = data.data
y = data.target

# Vieno karšto kodavimo paruošimas (transformuojame į dvinario formato kategorijas)
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# 2. Duomenų dalijimas į mokymo, validacijos ir testavimo rinkinius
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# 3. Duomenų standartizavimas
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 4. Funkcija treniruoti modelį su nurodytu optimizatoriumi ir loss funkcija
def train_with_optimizer(optimizer_name, loss_function):
    # Pasirinkite optimizatorių
    if optimizer_name == 'adam':
        optimizer = Adam()
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=0.01)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=0.01)
    elif optimizer_name == 'adagrad':
        optimizer = Adagrad(learning_rate=0.01)
    elif optimizer_name == 'adadelta':
        optimizer = Adadelta(learning_rate=0.01)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Sukuriame modelį
    model = Sequential([
        Input(shape=(X_train.shape[1],)),  # Pirmas sluoksnis kaip įvesties sluoksnis
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(y_train.shape[1], activation='softmax')
    ])

    # Kompiliuojame modelį
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    # Treniruojame modelį
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16, verbose=1)

    # Prognozuojame
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # Tikslumas
    acc = accuracy_score(y_test_labels, y_pred)
    return acc, history, y_pred, y_test_labels, model

# 5. Optimizatorių ir nuostolių funkcijų palyginimas ir mokymo istorijos vizualizavimas
optimizers = ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta']
loss_functions = ['categorical_crossentropy', 'huber_loss', 'mean_squared_error']  # Pridėjome mean_squared_error
results = {}
all_histories = {}
best_optimizer = None
best_accuracy = 0
best_predictions = None
best_labels = None
best_model = None
best_loss_function = None

# Kiekvienam optimizatoriui sukuriame atskirą figūrą
for opt in optimizers:
    plt.figure(figsize=(16, 8))  # Sukuriame naują figūrą kiekvienam optimizatoriui
    plt.suptitle(f'Optimizer: {opt.capitalize()}', fontsize=16)

    for loss_func in loss_functions:
        print(f"Training with optimizer: {opt} and loss function: {loss_func}")

        # Pasirenkame loss funkciją
        if loss_func == 'categorical_crossentropy':
            loss_function = 'categorical_crossentropy'
        elif loss_func == 'huber_loss':
            loss_function = Huber()  # Naudojame Huber loss
        elif loss_func == 'mean_squared_error':
            loss_function = MeanSquaredError()  # Naudojame MSE
        else:
            raise ValueError(f"Unknown loss function: {loss_func}")

        acc, history, y_pred, y_test_labels, model = train_with_optimizer(opt, loss_function)
        results[(opt, loss_func)] = acc
        all_histories[(opt, loss_func)] = history.history  # Išsaugome istoriją

        if acc > best_accuracy:
            best_accuracy = acc
            best_optimizer = opt
            best_loss_function = loss_func
            best_predictions = y_pred
            best_labels = y_test_labels
            best_model = model

        # Braižome praradimo (loss) grafiką
        plt.subplot(2, len(loss_functions), loss_functions.index(loss_func) + 1)
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.title(f'{loss_func.capitalize()} Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Braižome tikslumo (accuracy) grafiką
        plt.subplot(2, len(loss_functions), len(loss_functions) + loss_functions.index(loss_func) + 1)
        plt.plot(history.history['accuracy'], label='Training accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation accuracy')
        plt.title(f'{loss_func.capitalize()} Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

    # Rodyti visus grafikus
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Pakeičiam viršutinį tarpus, kad pavadinimas nebūtų užmaskuotas
    plt.show()

# 6. Rezultatų lentelė
print("Rezultatai su skirtingais optimizatoriais ir loss funkcijomis:")
for (opt, loss_func), acc in results.items():
    print(f'Optimizer: {opt.capitalize()}, Loss Function: {loss_func}, Test Accuracy: {acc:.4f}')

print(
    f'\nBest optimizer and loss function: {best_optimizer.capitalize()} with {best_loss_function}, Accuracy: {best_accuracy:.4f}')

# 7. Sumaišties matrica su geriausiu optimizatoriumi ir loss funkcija
cm = confusion_matrix(best_labels, best_predictions)

# Sumaišties matricos vizualizavimas
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.xlabel('Prognozės')
plt.ylabel('Tikrosios reikšmės')
plt.title(f'Confusion Matrix ({best_optimizer.capitalize()} + {best_loss_function})')
plt.show()
