import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor
interpreter = tflite.Interpreter(model_path='chessmen-model.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']
# Define classes
classes = ['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook']
# url = "https://thumbs.dreamstime.com/b/chess-flat-king-icon-stock-vector-image-royal-isolated-piece-outlined-214056020.jpg"
# Replace with your actual image URL
preprocessor = create_preprocessor('xception', target_size = (299, 299))
def predict(url):
    X_new = preprocessor.from_url(url)
    interpreter.set_tensor(input_index, X_new)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return dict(zip(classes, preds[0]))

def lambda_handler(event, context):
    url = event ['url']
    result = predict(url)
    return result