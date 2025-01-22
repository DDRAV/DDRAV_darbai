import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Duomenų įkėlimas ir paruošimas
digits = load_digits()
X, y = digits.images, digits.target

# Normalizuoti vaizdus ir konvertuoti į RGB
X = np.expand_dims(X, axis=-1)  # Pridėti kanalų dimensiją
X = np.tile(X, (1, 1, 1, 3))  # Konvertuoti į 3 kanalų vaizdą
X = X / 255.0  # Normalizacija

# Padalinti į treniravimo ir testavimo rinkinius
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Modeliai
vgg_model = VGG16(weights='imagenet')
vgg_feature_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('fc2').output)

resnet_model = ResNet50(weights='imagenet')
resnet_feature_model = Model(inputs=resnet_model.input, outputs=resnet_model.output)

# 3. Embeddings išgavimas
def extract_embeddings(model, X_data, target_size, preprocess_function):
    embeddings = []
    for image in X_data:
        # Keisti vaizdo dydį
        resized_image = tf.image.resize(image, target_size)
        image_array = preprocess_function(resized_image.numpy())
        image_array = np.expand_dims(image_array, axis=0)  # Pridėti partiją (batch dim)
        embedding = model.predict(image_array)
        embeddings.append(embedding.flatten())  # Flatten, jei reikia
    return np.array(embeddings)

# Išgauti embeddings su VGG16
vgg_embeddings_train = extract_embeddings(vgg_feature_model, X_train, (224, 224), preprocess_vgg)
vgg_embeddings_test = extract_embeddings(vgg_feature_model, X_test, (224, 224), preprocess_vgg)

# Išgauti embeddings su ResNet50
resnet_embeddings_train = extract_embeddings(resnet_feature_model, X_train, (224, 224), preprocess_resnet)
resnet_embeddings_test = extract_embeddings(resnet_feature_model, X_test, (224, 224), preprocess_resnet)

# 4. Klasifikacija
# Logistic Regression su VGG16 embeddings
clf_vgg = LogisticRegression(max_iter=1000)
clf_vgg.fit(vgg_embeddings_train, y_train)
y_pred_vgg = clf_vgg.predict(vgg_embeddings_test)
print("VGG16 Accuracy:", accuracy_score(y_test, y_pred_vgg))

# Logistic Regression su ResNet50 embeddings
clf_resnet = LogisticRegression(max_iter=1000)
clf_resnet.fit(resnet_embeddings_train, y_train)
y_pred_resnet = clf_resnet.predict(resnet_embeddings_test)
print("ResNet50 Accuracy:", accuracy_score(y_test, y_pred_resnet))
