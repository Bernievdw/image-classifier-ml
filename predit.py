# src/predict.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("model")

def predict_image(img_path, class_names, img_size=(224,224)):
    img = image.load_img(img_path, target_size=img_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0
    pred = model.predict(x)
    return class_names[np.argmax(pred)]

# Example
# class_names = ['cats', 'dogs']
# print(predict_image('dataset/test/cat1.jpg', class_names))
