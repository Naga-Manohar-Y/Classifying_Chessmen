FROM python:3.12-slim

RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["chessmen_predict.py","VGG19_v1_14_0.982.keras", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn","--bind=0.0.0.0:9696","chessmen_predict:app" ]