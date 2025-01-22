import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# 1. Duomenų paruošimas
# `load_digits` duomenys (8x8 pilkų atspalvių)
digits = load_digits()
X, y = digits.images, digits.target

# Konvertuoti į 3 kanalų vaizdus ir normalizuoti
X = np.expand_dims(X, axis=-1)  # Pridėti kanalų dimensiją
X = np.tile(X, (1, 1, 1, 3))  # Konvertuoti į RGB (3 kanalai)
X = X / 255.0  # Normalizuoti reikšmes į [0, 1]

# Padidinti vaizdus iki 224x224, kad tiktų VGG16 ir ResNet50
X_resized = np.array([tf.image.resize(img, (224, 224)).numpy() for img in X])

# Padalinti į treniravimo ir testavimo rinkinius
X_train, X_test, y_train, y_test = train_test_split(X_resized, y, test_size=0.2, random_state=42)

# 2. Modelių su embeddingų ištraukimu kūrimas
vgg_model = VGG16(weights='imagenet')
vgg_feature_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('fc2').output)

resnet_model = ResNet50(weights='imagenet')
resnet_feature_model = Model(inputs=resnet_model.input, outputs=resnet_model.get_layer('avg_pool').output)

# 3. MLP modelio kūrimas
def create_mlp_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))  # Mažiau neuronų
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# 4. Embeddingų generavimas ir treniravimas su MLP
def extract_and_train(model_feature, X_train, X_test, y_train, y_test, preprocess_function):
    # Preprocesuoti duomenis
    X_train_preprocessed = preprocess_function(X_train)
    X_test_preprocessed = preprocess_function(X_test)

    # Sukurti tf.data.Dataset su partijų dydžiu
    batch_size = 64
    train_dataset = tf.data.Dataset.from_tensor_slices(X_train_preprocessed).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(X_test_preprocessed).batch(batch_size)

    # Išgauti embeddingus partijomis
    embeddings_train = np.vstack([model_feature.predict(batch) for batch in train_dataset])
    embeddings_test = np.vstack([model_feature.predict(batch) for batch in test_dataset])

    # Sukurti ir treniruoti MLP modelį
    mlp_model = create_mlp_model(embeddings_train.shape[1])
    mlp_model.fit(embeddings_train, y_train, epochs=10, batch_size=batch_size, validation_split=0.2, verbose=1)

    # Tikslumo įvertinimas
    test_loss, test_acc = mlp_model.evaluate(embeddings_test, y_test, verbose=1)
    return test_acc

# 5. Tikslumo įvertinimas su VGG16
print("Evaluating with VGG16 embeddings...")
vgg_accuracy = extract_and_train(vgg_feature_model, X_train, X_test, y_train, y_test, preprocess_vgg)
print(f'VGG16 MLP Test accuracy: {vgg_accuracy:.4f}')

# 6. Tikslumo įvertinimas su ResNet50
print("Evaluating with ResNet50 embeddings...")
resnet_accuracy = extract_and_train(resnet_feature_model, X_train, X_test, y_train, y_test, preprocess_resnet)
print(f'ResNet50 MLP Test accuracy: {resnet_accuracy:.4f}')

# 7. Analizė ant naudotojo nuotraukos
def preprocess_image(image_path, target_size, preprocess_function):
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image)
    image_array = preprocess_function(image_array)
    return tf.expand_dims(image_array, axis=0)

# Kelias į naudotojo nuotrauką
image_path = 'flowers.jpg'  # Nurodykite tikrą kelią į paveikslėlį

# Analizuojame naudotojo nuotrauką su VGG16
user_image_vgg_embedding = vgg_feature_model.predict(preprocess_image(image_path, (224, 224), preprocess_vgg))
print(f"User image VGG16 embedding shape: {user_image_vgg_embedding.shape}")

# Analizuojame naudotojo nuotrauką su ResNet50
user_image_resnet_embedding = resnet_feature_model.predict(preprocess_image(image_path, (224, 224), preprocess_resnet))
print(f"User image ResNet50 embedding shape: {user_image_resnet_embedding.shape}")

# Tikriname naudotojo nuotraukos embeddingus su MLP (Digits)
def classify_user_image(user_embedding, mlp_model):
    prediction = mlp_model.predict(user_embedding)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class

# Naudotojo nuotraukos klasifikacija (su digits duomenimis treniruotu MLP)
vgg_digits_mlp_model = create_mlp_model(user_image_vgg_embedding.shape[1])
resnet_digits_mlp_model = create_mlp_model(user_image_resnet_embedding.shape[1])

# Treniruoti modelius
vgg_digits_mlp_model.fit(vgg_feature_model.predict(X_train), y_train, epochs=10, batch_size=64, verbose=1)
resnet_digits_mlp_model.fit(resnet_feature_model.predict(X_train), y_train, epochs=10, batch_size=64, verbose=1)

# Klasifikuoti naudotojo nuotrauką
vgg_prediction = classify_user_image(user_image_vgg_embedding, vgg_digits_mlp_model)
resnet_prediction = classify_user_image(user_image_resnet_embedding, resnet_digits_mlp_model)

print(f"VGG16-based MLP prediction for user image: {vgg_prediction}")
print(f"ResNet50-based MLP prediction for user image: {resnet_prediction}")
