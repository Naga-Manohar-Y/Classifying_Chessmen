import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Configure page layout
st.set_page_config(layout="wide")

# Title and description
st.write("""
# ♟️ Chess Piece Prediction - Classification
This app predicts the class of a chess piece using a deep learning model.
""")

st.write("### Enter the URL of your image:")

# Input: URL
url = st.text_input(
    "Enter your image URL here 👇",
    placeholder="https://...",
    label_visibility="visible"
)

# Columns for display
col1, col2 = st.columns(2)

# Function to fetch and preprocess the image
def fetch_image(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    except Exception as e:
        st.error(f"Error fetching the image: {e}")
        return None

# Function to send the image to the prediction API
def predict_image(image_url):
    api_url = 'https://w7ec8wrj9k.execute-api.us-east-1.amazonaws.com/chessman/predict'
    data = {'url': image_url}
    try:
        response = requests.post(api_url, json=data)
        response.raise_for_status()  # Raise HTTPError for bad responses
        return response.json()
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Handle URL input
if url:
    # Fetch the image
    image = fetch_image(url)
    if image:
        with col1:
            st.header("Input Image")
            st.image(image, caption="Uploaded Image (Smaller Size)", width=200)

        # Call prediction API
        result = predict_image(url)
        if result:
            predicted_class = max(result, key=result.get)
            probability = result[predicted_class]
            with col2:
                st.header("Prediction Result")
                st.write(f"**Predicted Class:** {predicted_class}")
                st.write(f"**Probability:** {probability:.4f}")
