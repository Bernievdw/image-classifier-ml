from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def build_simple_cnn(num_classes, img_size=(224,224)):
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(img_size[0], img_size[1], 3)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_mobilenet(num_classes, img_size=(224,224)):
    base_model = MobileNetV2(input_shape=(img_size[0], img_size[1],3), include_top=False, weights='imagenet')
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
