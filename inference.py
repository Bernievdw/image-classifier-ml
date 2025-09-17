import os
from PIL import Image
import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

IMG_SIZE = 224
DEFAULT_MODEL_PATH = os.environ.get('MODEL_PATH', 'model/saved_model')
_model = None

class MockModel:
    """Simple deterministic mock model used when TF isn't installed."""
    def predict(self, x):
        return np.array([[0.7]])

def load_model(path='model/saved_model'):
    if not os.path.exists(path):
        print("Model not found, using mock model for testing...")
        class MockModel:
            def predict(self, x):
                return np.array([[0.7]])
        return MockModel()
    import tensorflow as tf
    return tf.keras.models.load_model(path)

def load_and_preprocess_image(path_or_pil, img_size=IMG_SIZE):
    if isinstance(path_or_pil, str):
        img = Image.open(path_or_pil).convert('RGB')
    else:
        img = path_or_pil.convert('RGB')
    img = img.resize((img_size, img_size))
    arr = np.array(img).astype('float32')
    if TF_AVAILABLE:
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    else:
        arr = arr / 127.5 - 1.0
    arr = np.expand_dims(arr, 0)
    return arr

def predict_from_path(path, model_path=DEFAULT_MODEL_PATH):
    model = load_model(model_path)
    x = load_and_preprocess_image(path)
    score = float(model.predict(x)[0][0])
    label = 'dog' if score >= 0.5 else 'cat'
    confidence = score if score >= 0.5 else (1.0 - score)
    return {'label': label, 'confidence': float(confidence), 'raw_score': float(score)}

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python inference.py /path/to/image.jpg')
        sys.exit(1)
    print(predict_from_path(sys.argv[1]))
