FROM public.ecr.aws/lambda/python:3.10
RUN pip install keras-image-helper
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
# Copy necessary files
COPY ["lambda_function.py", "chessmen-model.tflite"]
# Specify the entry point
CMD [ "lambda_function.lambda_handler"]