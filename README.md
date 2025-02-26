# Chessmen Classification Using Deep Learning
This project aims to classify chess pieces (e.g., King, Queen, Bishop, etc.) using a deep learning model. The solution involves data preprocessing, model training, and deploying the trained model as a Flask web service. Finally, the model is deployed on AWS using Lambda, ECR, and API Gateway.

![App](https://github.com/Naga-Manohar-Y/Classifying_Chessmen/blob/main/Project_Media/Chessman.gif)


## Problem Description

Chess is a game of strategy, precision, and intellectual mastery. Accurately identifying chess pieces from images is a step toward automating chess game analysis, enhancing digital chess applications, and providing innovative solutions for enthusiasts and developers. This project aims to classify chess pieces—King, Queen, Rook, Bishop, Knight, and Pawn—using advanced machine learning techniques with a focus on accuracy and deployment readiness.

---
## Architecture Overview
![Architecture Diagram](https://github.com/Naga-Manohar-Y/Classifying_Chessmen/blob/main/Project_Media/chessman_architecture.png)
---
## Project Workflow

### 1. **Exploratory Data Analysis (EDA)**
EDA was conducted to understand the dataset better, including:
- **Dataset Overview:** This dataset is structured into six directories, one for each chess piece. Each subfolder contains labeled images of the respective chess piece.
- **Source:** [Kaggle Chessman Dataset](https://www.kaggle.com/datasets/niteshfre/chessman-image-dataset/data)
  - You can directly use the images from this repo before running the `chessmen.ipynb` notebook change the `path` or `DIR` variable to 'Chessman_Images_Data'
- **Visualizations:** Sample images of each chess piece and their respective pixel distributions. Class distribution, total samples, and data augmentation techniques are displayed in  `chessmen.ipynb` file.
- **Insights:** Key observation that influenced preprocessing and model architecture decisions is less number of images (600 images).
  - So I applied data augmentation which helps the model to train on different variations of images.

### 2. **Model Training**
- Model Architectures used:
  - **VGG19**
  - **MobileNet**
  - **ResNet50**
  - **Xception**
- Training Process:
  - Utilized pre-trained model on ImageNet and performed exceptionally well on this dataset.
  - Performed parameter tuning with learning rate, dropout, and fine-tuning last k layers.
  - Fine-tuned  last k layers in the model architecture on chess image data by unfreezing the layers.
- Optimizer: Adam.
- Evaluation metric: Accuracy.
- Training/Validation Split: 80/20 ratio.
- Results: VGG19 outperformed other models due to its simplicity and efficiency in handling this dataset.
- Finally trained the larger model (**image_size - 299 x 299**) in `train.py` which gave **98.2%** accuracy on validation_data.

### 3. **Exporting Notebook to Script**
- To streamline deployment, the Jupyter Notebook was converted into a Python script. This ensures reproducibility and simplifies integration into the deployment pipeline.
- `train.py` for training the model
- `predict.py` for local inference
- `chessmen_predict.py` for inference from web service using Flask
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
  git clone https://github.com/Naga-Manohar-Y/Classifying_Chessmen.git
  cd Classifying_Chessmen
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
  
  Now choose the saved model for example `VGG19_v1_14_0.982.keras` and update the model name wherever we are loading (`chessmen_predict.py`) and copying it into docker container.
  
  ```
  chessmen_predict.py:

  print("Loading the model...")
  VGG19_model = keras.models.load_model('./VGG19_v1_14_0.982.keras')
  print("Model loaded successfully.")

  Dockerfile:

  COPY ["chessmen_predict.py","VGG19_v1_14_0.982.keras", "./"]

  ```


4. **Build the Docker Image**:

  After ensuring that the project files are in place, build the Docker image by running the following command in the project directory:
  
  ```bash
  docker build -t chessmen:latest .
  ```
5. **Run the Docker Container**:

  Once the Docker image is built, you can run the container:
  ```bash
  docker run -p 9696:9696 chessmen:latest
  ```
  This will run the application inside a Docker container and map port 9696 from the container to your local machine.

6. **Utilize the model as a web service:**
  ```bash
  import requests

  service_url = 'http://localhost:9696/predict'
  data = {"url": "https://thumbs.dreamstime.com/b/chess-flat-king-icon-stock-vector-image-royal-isolated-piece- outlined-214056020.jpg"}

  response = requests.post(service_url, json=data)
  data = response.json()
  data['predicted_class']
  ```
  Output:
  ```bash
  {"predicted_class":"King",
  "probabilities":{"Bishop":0.0023368641268461943,
                 "King":0.9764311909675598,
                 "Knight":0.0006968219531700015,
                 "Pawn":0.0013725616736337543,
                 "Queen":0.01652485504746437,
                 "Rook":0.0026377495378255844
                 }
  }
  ```

## ☁️ Cloud Deployment
**Refer [cloud_deploy](https://github.com/Naga-Manohar-Y/Classifying_Chessmen/tree/main/cloud_deploy) folder.**

1. **Converting the TensorFlow Model to TFLite**

- Convert your TensorFlow model generated after running `train.py` into a TensorFlow Lite (TFLite) model.
- Refer to the `tflite_model.py` script, where the conversion is implemented, and inference is tested.
- Once the model is converted, save it as `chessmen-model.tflite`. This model will be used for cloud deployment.

2. **Building the Docker Image for AWS Lambda**

- Create a `Dockerfile` with the following content:

  ```Dockerfile
  # Use AWS Lambda Python 3.10 base image
  FROM public.ecr.aws/lambda/python:3.10
  
  # Install necessary Python libraries
  RUN pip install keras-image-helper
  RUN pip install numpy==1.23.1
  RUN pip install --no-deps https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
  
  # Copy the TensorFlow Lite model and Lambda function code into the image
  COPY "chessmen-model.tflite" .
  COPY "lambda_function.py" .
  
  # Specify the entry point for the AWS Lambda function
  CMD ["lambda_function.lambda_handler"]
  ```
- For M1/M2 users build the image with --platform=linux/amd64 for compatibility:
  ```bash
  docker build --platform=linux/amd64 -t chessmen-tflite .
  ```
- For Other Systems
Use the standard build command:
  ```bash
  docker build -t chessmen-tflite .
  ```
- List the Docker images to confirm the build was successful:
  ```bash
  docker images
  ```
- Run the container, mapping the port to 8081 (or any available port):
  ```bash
  docker run -it --rm -p 8081:8080 chessmen-tflite
  ```
- Use the local_test.py script to test the container locally. The script should send an HTTP POST request to the running container at http://localhost:8081:
  ```bash
  python local_test.py
  ```

3. **Publishing the image to AWS ECR**
- Install AWS CLI
  ```bash
  pip install awscli
  ```
- Create a new repository in AWS ECR:
  ```bash
  aws ecr create-repository --repository-name chessman-tflite-image
  ```
- After creating the repository, configure the environment variables:
  ```bash
  ACCOUNT=481665108850
  REGION=us-east-1
  REGISTRY=chessman-tflite-image
  PREFIX=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REGISTRY}
  TAG=chessman-model-vgg-v1
  REMOTE_URI=${PREFIX}:${TAG}
  
  echo ${REMOTE_URI}
  ```
- Tag the locally built Docker image:
  ```bash
  docker tag chessmen-tflite:latest ${REMOTE_URI}
  ```
- Push the image to the AWS ECR repository:
  ```bash
  docker push ${REMOTE_URI}
  ```
  ![ECR](https://github.com/Naga-Manohar-Y/Classifying_Chessmen/blob/main/Project_Media/ECR.png)

- Create a Lambda Function with the ECR Image
  - Open the AWS Lambda console.
  - Choose "Create function" and select the "Container image" option.
  - Select the image from the ECR repository (chessman-model-vgg-v1) and deploy it.
- Test the Lambda Function
  - Update the basic settings for the Lambda function:
    Memory: 512 MB
    Timeout: 30 seconds
  - Create and save a test event with the required input data. For example:
    ```bash
    {
      "url": "https://example.com/image.jpg"
    }
    ```
  - Run the test. Verify that the Lambda function executes successfully and returns the expected output.
    ![Lambda_function](https://github.com/Naga-Manohar-Y/Classifying_Chessmen/blob/main/Project_Media/Lambda_function.png)
- Expose the Lambda Function via API Gateway
  - Create a New REST API
  - Go to the API Gateway console.
  - Choose "Create API" > "REST API".
  - Create a new resource for the API.
  - Create a POST Method
  - Under the created resource, add a new method with the POST type.
  - Link the method to the created Lambda function.
  - Deploy the API by creating a stage (e.g., chessman).
    ![API_Gateway](https://github.com/Naga-Manohar-Y/Classifying_Chessmen/blob/main/Project_Media/API_Gateway.png)
- Test the Lambda Function via API Gateway
  - Run the python script (cloud_test.py) to test the deployed API Gateway endpoint.
    ![Test](https://github.com/Naga-Manohar-Y/Classifying_Chessmen/blob/main/Project_Media/Cloud_Test%20copy.png)

4. **Streamlit App**
- Install streamlit
  ```bash
  pip install streamlit
  ```
- Run the file chessman_app.py
  ```bash
  cd cloud_deploy
  python chessman_app.py
  ```
- Streamlit Interface: User-friendly interface for seamless interaction.
- Image URL Input: Users can input the URL of a chess piece image.
- Image Fetching: Fetches and displays the input image for verification.
- Prediction API Integration: Sends the image to a cloud-hosted prediction API (AWS Lambda) for classification.
- Prediction Results: Displays the predicted class and its probability with a clear and concise layout.

