import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.models import Model
import numpy as np

# 1. Paveikslėlio ir embeddingų paruošimas
def preprocess_and_extract_embeddings(image, model, preprocess_function, target_size=(224, 224)):
    image_resized = tf.image.resize(image, target_size)  # Keisti dydį
    image_array = preprocess_function(image_resized.numpy())  # Normalizacija
    image_array = np.expand_dims(image_array, axis=0)  # Pridėti partiją
    return model.predict(image_array)

# 2. `load_digits` rinkinio paruošimas
digits = load_digits()
X, y = digits.images, digits.target
X = np.expand_dims(X, axis=-1)  # Pridėti kanalų dimensiją
X = np.tile(X, (1, 1, 1, 3))  # Konvertuoti į 3 kanalų vaizdą
X = X / 255.0  # Normalizacija

# Padalinti į treniravimo ir testavimo rinkinius
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Išgauti embeddingus su VGG16 ir ResNet50
vgg_model = VGG16(weights='imagenet')
vgg_feature_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('fc2').output)

resnet_model = ResNet50(weights='imagenet')
resnet_feature_model = Model(inputs=resnet_model.input, outputs=resnet_model.get_layer('avg_pool').output)

# 4. MLP modelio kūrimas
def create_mlp_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))  # 10 klasių, pagal `digits` duomenų rinkinį
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 5. Išgauti embeddingus ir treniruoti su MLP
def extract_and_train(model_feature, X_train, X_test, y_train, y_test, preprocess_function):
    # Išgauti embeddingus
    embeddings_train = np.array([embedding.flatten() for embedding in [preprocess_and_extract_embeddings(x, model_feature, preprocess_function) for x in X_train]])
    embeddings_test = np.array([embedding.flatten() for embedding in [preprocess_and_extract_embeddings(x, model_feature, preprocess_function) for x in X_test]])

    # Sukurti ir treniruoti MLP modelį
    mlp_model = create_mlp_model(embeddings_train.shape[1])
    mlp_model.fit(embeddings_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

    # Tikslumo įvertinimas
    test_loss, test_acc = mlp_model.evaluate(embeddings_test, y_test)
    return test_acc

# 6. Tikslumo įvertinimas su VGG16
print("Evaluating with VGG16 embeddings...")
vgg_accuracy = extract_and_train(vgg_feature_model, X_train, X_test, y_train, y_test, preprocess_vgg)
print(f'VGG16 MLP Test accuracy: {vgg_accuracy:.4f}')

# 7. Tikslumo įvertinimas su ResNet50
print("Evaluating with ResNet50 embeddings...")
resnet_accuracy = extract_and_train(resnet_feature_model, X_train, X_test, y_train, y_test, preprocess_resnet)
print(f'ResNet50 MLP Test accuracy: {resnet_accuracy:.4f}')

# 8. Analizuoti konkretų paveiksliuką (pvz., `flowers.jpg`)
def preprocess_image(image_path, target_size, preprocess_function):
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image)
    image_array = preprocess_function(image_array)
    return tf.expand_dims(image_array, axis=0)

# Kelias į nuotrauką
image_path = 'flowers.jpg'  # Nurodykite tikrą kelią į paveikslėlį

# Analizuojame su VGG16
vgg_embedding = preprocess_and_extract_embeddings(preprocess_image(image_path, (224, 224), preprocess_vgg), vgg_feature_model, preprocess_vgg)
print(f"VGG16 embedding for flowers image: {vgg_embedding}")

# Analizuojame su ResNet50
resnet_embedding = preprocess_and_extract_embeddings(preprocess_image(image_path, (224, 224), preprocess_resnet), resnet_feature_model, preprocess_resnet)
print(f"ResNet50 embedding for flowers image: {resnet_embedding}")
