# Classifying_Chessmen
This project aims to classify chess pieces (e.g., King, Queen, Bishop, etc.) using a deep learning model. The solution involves data preprocessing, model training, and deploying the trained model as a Flask web service.

## Problem Description

Chess is a game of strategy, precision, and intellectual mastery. Accurately identifying chess pieces from images is a step toward automating chess game analysis, enhancing digital chess applications, and providing innovative solutions for enthusiasts and developers. This project aims to classify chess pieces—King, Queen, Rook, Bishop, Knight, and Pawn—using advanced machine learning techniques with a focus on accuracy and deployment readiness.

---

## Project Workflow

### 1. **Exploratory Data Analysis (EDA)**
EDA was conducted to understand the dataset better, including:
- **Dataset Overview:** This dataset is structured into six directories, one for each chess piece. Each subfolder contains labeled images of the respective chess piece.
- **Source:** [Kaggle Chessman Dataset](https://www.kaggle.com/datasets/niteshfre/chessman-image-dataset/data)
- Class distribution, total samples, and data augmentation techniques are there is notebook file.
- **Visualizations:** Sample images of each chess piece and their respective pixel distributions.
- **Insights:** Key observations that influenced preprocessing and model architecture decisions.

### 2. **Model Training**
- Model Architectures used:
  - **VGG19**
  - **MobileNet**
  - **ResNet50**
  - **Xception**
- Training Process:
  - Fine-tuned with adjusted learning rate, dropout, and selective unfreezing of layers.
  - Pre-trained on ImageNet and performed exceptionally well on this dataset.
- Optimizer: Adam.
- Evaluation metric: Accuracy.
- Training/Validation Split: 80/20 ratio.
- Results: VGG19 outperformed other models due to its simplicity and efficiency in handling this dataset.
- Finally trained the larger model (image_size - 299 x 299) in train.py which gave 98.2% accuracy on validation_data.

### 3. **Exporting Notebook to Script**
- To streamline deployment, the Jupyter Notebook was converted into a Python script. This ensures reproducibility and simplifies integration into the deployment pipeline.
- `train.py` for training the model
- `predict.py` for inference from web service
- `chessmen_predict.py` for inference from web service
- Note: If you want try tensorflowlite model refer to `tflite_model.py`

### 4. **Model Deployment**
The trained model was deployed as a RESTful API using Flask and Gunicorn. The service allows users to:

- Send an image URL to the endpoint.
- Receive predictions, including the class label and probabilities for all classes.
**API Endpoint**:

- POST /predict
  - Input: JSON object with the image URL.
  - Output: Predicted class and probabilities.
```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"url": "https://example.com/chess_piece.jpg"}' \
http://localhost:9696/predict

```

### 5. **Reproducibility**
- Reproducibility is ensured by using clear scripts (`train.py` and `chessmen_predict.py`), specifying dependencies in `Pipfile`, and documenting each step.
- Application is containerized using Docker for consistency and scalability.

---

## Dependency and Environment Management

- **Pipfile** and **Pipfile.lock** are used to define and lock the Python dependencies for the project.
- **Key Dependencies**:
  - tensorflow
  - keras
  - Flask
  - pandas
  - numpy
  - pillow

---

## Containerization

- A **Dockerfile** is provided for containerizing the application. The Docker container includes the following:
  - Installation of required dependencies using `pipenv`.
  - Configuration of the Flask app for serving predictions.
  - Exposing port `9696` for the web service.

---

## How to Run It

### Prerequisites
- Python 3.12+
- Docker (for containerized deployment)

### Steps to Run Locally
1. **Clone the Repository**:
```bash
git clone https://github.com/Naga-Manohar-Y/Chessmen_Classification.git
cd Chessmen_Classification
```
2. **Install Dependencies**:

```bash
pip install pipenv
pipenv install
```
This will create a virtual environment and install all required dependencies for the project.

3. **Train the Model**:

```bash
python train.py
```
This script will train the model using the training dataset, and save the models with the help of checkpointing, which can then be used for predictions.

4. **Build the Docker Image**:

After ensuring that the project files are in place, build the Docker image by running the following command in the project directory:
```bash
docker build -t chessmen:latest .
```
5. **Run the Docker Container**:

Once the Docker image is built, you can run the container:
```bash
docker run -p 9696:9696 chessmen:latest
This will run the application inside a Docker container and map port 9696 from the container to your local machine.
```

