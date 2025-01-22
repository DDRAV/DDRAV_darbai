from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import decode_predictions as predict_vgg
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.resnet50 import decode_predictions as predict_resnet
from tensorflow.keras.models import Model
import tensorflow as tf

# Kelias į nuotrauką
image_path = 'flowers.jpg'

# 1. VGG16 modelis
vgg_model = VGG16(weights='imagenet')
vgg_feature_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('fc2').output)

# Paruošti vaizdą VGG16 modeliui
image_vgg = load_img(image_path, target_size=(224, 224))
image_array_vgg = img_to_array(image_vgg)
image_array_vgg = preprocess_vgg(image_array_vgg)
image_array_vgg = tf.expand_dims(image_array_vgg, axis=0)

# Prognozė ir embedding išgavimas su VGG16
predictions_vgg = vgg_model.predict(image_array_vgg)
decoded_predictions_vgg = predict_vgg(predictions_vgg, top=3)
print("\nVGG16 Predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions_vgg[0]):
    print(f"{i + 1}: {label} ({score:.2f})")

vgg_embeddings = vgg_feature_model.predict(image_array_vgg)
print("VGG16 Embeddings Shape:", vgg_embeddings.shape)

# 2. ResNet50 modelis
resnet_model = ResNet50(weights='imagenet')
resnet_feature_model = Model(inputs=resnet_model.input, outputs=resnet_model.output)

# Paruošti vaizdą ResNet50 modeliui
image_resnet = load_img(image_path, target_size=(224, 224))
image_array_resnet = img_to_array(image_resnet)
image_array_resnet = preprocess_resnet(image_array_resnet)
image_array_resnet = tf.expand_dims(image_array_resnet, axis=0)

# Prognozė su ResNet50
predictions_resnet = resnet_model.predict(image_array_resnet)
decoded_predictions_resnet = predict_resnet(predictions_resnet, top=3)
print("\nResNet50 Predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions_resnet[0]):
    print(f"{i + 1}: {label} ({score:.2f})")

resnet_embeddings = resnet_feature_model.predict(image_array_resnet)
print("ResNet50 Embeddings Shape:", resnet_embeddings.shape)