import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input
import tensorflow.lite as tflite

import requests
import io
from PIL import Image
import numpy as np
import keras
from tensorflow.keras.applications.xception import preprocess_input


model = keras.models.load_model('./VGG19_v1_14_0.982.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('chessmen-model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)

interpreter = tflite.Interpreter(model_path='chessmen-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# Define classes
classes = ['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook']

def load_image_from_url(url):
    """
    Load an image from the provided URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Check if the URL points to an image
        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type:
            raise ValueError(f"URL does not point to an image. Content-Type: {content_type}")

        # Open the image
        img = Image.open(io.BytesIO(response.content))
        return img
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")

def pred_chess(url):

    image = load_image_from_url(url)

    img = image.resize((299, 299), Image.NEAREST)

    x = np.array(img)
    X = np.array([x])
    print(X.shape)
    X = preprocess_input(X)
    return X

url = "https://thumbs.dreamstime.com/b/chess-flat-king-icon-stock-vector-image-royal-isolated-piece-outlined-214056020.jpg"  # Replace with your actual image URL


X = pred_chess(url)

interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)

dict(zip(classes, preds[0]))
