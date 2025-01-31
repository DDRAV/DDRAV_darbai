import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Duomenų įkėlimas
data = load_digits()
X = data.data
y = data.target

# Vienkarštis kodavimas (One-Hot Encoding) klasifikacijos tikslams
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Duomenų padalinimas į mokymo, validacijos ir testavimo rinkinius
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Duomenų normalizavimas
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Modelio kūrimas
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # Didesnis sluoksnis dėl didesnės įvesties
    Dropout(0.3),  # Reguliarizacija
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax')  # Softmax aktyvacija kelių klasių klasifikacijai
])

# Mokymosi greičio mažinimo schema
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=50,
    decay_rate=0.95
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Modelio kompiliavimas
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Modelio mokymas
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[early_stopping]
)

# Grafikai
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss during training')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy during training')
plt.legend()

plt.show()

# Testavimo analizė
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Tikslumo skaičiavimas
acc = accuracy_score(y_test_labels, y_pred)
print('Test Accuracy: ', acc)

# Painiavos matrica
cm = confusion_matrix(y_test_labels, y_pred)
print('Confusion Matrix:\n', cm)
