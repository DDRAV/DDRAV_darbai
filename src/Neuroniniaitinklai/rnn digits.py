import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Patikriname TensorFlow versiją ir *eager execution* būseną
print("TensorFlow version:", tf.__version__)
print("Eager execution enabled:", tf.executing_eagerly())

# Duomenų įkėlimas
digits = load_digits()
X = digits.images
y = digits.target

# Duomenų paruošimas
scaler = StandardScaler()
X_flat = X.reshape(X.shape[0], -1)
X_scaled = scaler.fit_transform(X_flat)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 8, 8))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Konvertuojame į NumPy masyvus
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

# Hiperparametrai
batch_sizes = [16, 32, 64]
optimizers = {
    "Adam": tf.keras.optimizers.Adam(),
    "SGD": tf.keras.optimizers.SGD(),
    "RMSprop": tf.keras.optimizers.RMSprop()
}
loss_functions = ["sparse_categorical_crossentropy", "mean_squared_error"]

results = {}

# Modelių treniravimas
for opt_name, optimizer in optimizers.items():
    for loss in loss_functions:
        for batch_size in batch_sizes:
            print(f"Training with Optimizer: {opt_name}, Loss: {loss}, Batch Size: {batch_size}")

            # Sukuriamas modelis
            inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
            x = tf.keras.layers.SimpleRNN(64, activation='relu')(inputs)
            outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
            model = tf.keras.Model(inputs, outputs)

            # Kompiliavimas
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

            # Mokymas
            try:
                history = model.fit(
                    X_train, y_train,
                    epochs=20, batch_size=batch_size,
                    validation_data=(X_test, y_test),
                    verbose=1
                )
            except Exception as e:
                print(f"Error with Optimizer: {opt_name}, Loss: {loss}, Batch Size: {batch_size}")
                print(e)
                continue

            # Rezultatų saugojimas su unikaliu raktu
            key = f"{opt_name}_loss-{loss}_batch-{batch_size}"
            results[key] = {
                "val_accuracy": history.history['val_accuracy']
            }

# Rezultatų vizualizacija: 2 grafikai viename paveikslėlyje
for opt_name in optimizers.keys():
    for loss in loss_functions:
        fig, ax = plt.subplots(figsize=(10, 6))  # One plot for accuracy

        ax.set_title(f"Validation Accuracy for {opt_name} - {loss}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Validation Accuracy")

        for batch_size in batch_sizes:  # Iterate through all batch sizes
            key = f"{opt_name}_loss-{loss}_batch-{batch_size}"
            if key in results:
                ax.plot(results[key]["val_accuracy"], label=f"Batch {batch_size}")

        ax.legend()
        ax.grid()
        plt.tight_layout()
        plt.show()
