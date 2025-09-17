import tempfile
from PIL import Image
import numpy as np
from inference import load_and_preprocess_image

def test_load_and_preprocess_image():
    img = Image.fromarray((np.random.rand(100, 100, 3) * 255).astype('uint8'))
    with tempfile.NamedTemporaryFile(suffix='.jpg') as f:
        img.save(f, format='JPEG')
        f.flush()
        arr = load_and_preprocess_image(f.name)
        assert arr.shape == (1, 224, 224, 3)
        assert arr.dtype in ('float32', 'float64')
