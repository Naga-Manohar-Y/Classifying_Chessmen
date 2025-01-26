import requests

url = 'http://localhost:8081/2015-03-31/functions/function/invocations'


data = {'url' : 'https://thumbs.dreamstime.com/b/chess-flat-king-icon-stock-vector-image-royal-isolated-piece-outlined-214056020.jpg'}


# Sending a POST request to the Lambda function
result = requests.post(url, json=data).json()

# Printing the prediction result
print(result)