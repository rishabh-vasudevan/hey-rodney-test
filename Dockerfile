#Dockerfile

from python:3.8-buster

WORKDIR /app

WORKDIR /run

ADD processing_and_training.py .

ADD requirements.txt .

ADD break_background_wav_to_1_sec.py .

ADD run.sh .

RUN chmod a+x run.sh

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc libsndfile1 

RUN apt-get install -y libasound-dev -y

RUN apt-get install libportaudio2 -y

RUN pip install -r requirements.txt

CMD ["./run.sh"]