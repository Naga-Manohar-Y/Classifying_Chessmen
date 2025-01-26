

import requests

url = 'https://w7ec8wrj9k.execute-api.us-east-1.amazonaws.com/chessman/predict'


data = {'url' : 'https://thumbs.dreamstime.com/b/chess-flat-king-icon-stock-vector-image-royal-isolated-piece-outlined-214056020.jpg'}


# Sending a POST request to the Lambda function
result = requests.post(url, json=data).json()

predicted_class = max(result, key=result.get)

# Printing the predicted class and its probability
print(f"Predicted class: {predicted_class}, Probability: {result[predicted_class]:.4f}")

'''
other urls you can try
https://thumbs.dreamstime.com/z/bishop-chess-piece-wooden-isolated-clipping-path-included-43949803.jpg

https://images.chesscomfiles.com/uploads/v1/article/15680.8ba56ba3.5000x5000o.1c5591a9bd76.jpeg
'''
