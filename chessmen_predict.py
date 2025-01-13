import requests
import io
from PIL import Image
import numpy as np
import keras
from tensorflow.keras.applications.xception import preprocess_input
from flask import Flask, request, jsonify

# Load the trained model
print("Loading the model...")
VGG19_model = keras.models.load_model('./VGG19_v1_14_0.982.keras')
print("Model loaded successfully.")

# Define classes
classes = ['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook']

# Flask app
app = Flask("ChessmenClassifier")

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

def predict_chess_piece(url):
    """
    Predict the chess piece from the image at the provided URL.
    """
    try:
        # Load and preprocess the image
        image = load_image_from_url(url)
        image = image.resize((299, 299), Image.NEAREST)

        x = np.array(image)
        X = np.array([x])
        X = preprocess_input(X)

        # Predict using the loaded model
        pred = VGG19_model.predict(X)
        #class_probabilities = dict(zip(classes, pred[0]))
        class_probabilities = {classes[i]: float(pred[0][i]) for i in range(len(classes))}


        # Get the class with the highest probability
        predicted_class = max(class_probabilities, key=class_probabilities.get)

        return {
            'predicted_class': predicted_class,
            'probabilities': class_probabilities
        }
    except Exception as e:
        return {'error': str(e)}

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict the chess piece.
    """
    try:
        data = request.get_json()
        url = data.get('url')

        if not url:
            return jsonify({'error': 'URL not provided'}), 400

        # Call the prediction function
        result = predict_chess_piece(url)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
