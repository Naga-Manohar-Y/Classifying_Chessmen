import requests
import io
from PIL import Image
import numpy as np
import keras
from tensorflow.keras.applications.xception import preprocess_input

print("required libraries are imported")

print("*******************************************************************")

def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Check Content-Type header
        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type:
            raise ValueError(f"URL does not point to an image. Content-Type: {content_type}")

        # Attempt to open the image
        img = Image.open(io.BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

classes = ['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook']

def pred_chess(url):

    image = load_image_from_url(url)
    
    print("image loaded from the url")

    img = image.resize((299, 299), Image.NEAREST)

    x = np.array(img)
    X = np.array([x])
    print(X.shape)
    X = preprocess_input(X)
    pred = VGG19_model.predict(X)

    class_probabilities = dict(zip(classes, pred[0]))

    predicted_class = max(class_probabilities, key=class_probabilities.get)

    return predicted_class

print("Loading the model")

VGG19_model = keras.models.load_model('./VGG19_v1_02_0.578.keras')

print("*******************************************************************")


url = "https://thumbs.dreamstime.com/b/chess-flat-king-icon-stock-vector-image-royal-isolated-piece-outlined-214056020.jpg"  

print(pred_chess(url))