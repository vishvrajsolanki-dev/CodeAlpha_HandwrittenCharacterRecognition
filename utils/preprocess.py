import numpy as np
from PIL import Image, ImageOps

def preprocess_canvas(img_data):
    img = Image.fromarray(img_data.astype("uint8"), "RGBA").convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    return arr.reshape(1, 28, 28, 1)

def preprocess_upload(uploaded_file):
    img = Image.open(uploaded_file).convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    return arr.reshape(1, 28, 28, 1)