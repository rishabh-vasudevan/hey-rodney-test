#Dockerfile

from python:3.8-buster

WORKDIR /app

WORKDIR /run

ADD processing_and_training.py .

ADD requirements.txt .

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc libsndfile1 

RUN pip install -r requirements.txt

CMD ["python", "/run/processing_and_training.py"]