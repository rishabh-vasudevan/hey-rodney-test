# Train a wake word model

## About

  This project allows you to make a wake word detection model. This repository contains scripts to train the model and also run inference/prediction. Implementation of live audio classification is also available

## Train the wake word model

### Run the training with Docker

- Clone the repository

- Build the docker image using the command

```
docker build -t processing-and-training -f Dockerfile.training .
```
 
- Mount the folder which contains the `speech_command_dataset` folder to the `/app` folder in the docker contianer
- It will breakdown the background noises to 1 sec long wav files if it is not already broken down
- It will ask for the wake word you want to train on, you can input the wake word here ( Make sure it is the same as one of the directories present in the speech command dataset, if your preffered wake word is not present then follow instructions given below to add word directory to the speech_command_dataset )
- To run the docker container write the following command

```
docker run -it -v "$(pwd)":/app processing-and-training
```


### Run the training without Docker
- Clone the repository

- Download the Speech Command dataset folder from : https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz 

- Make a new folder with the name `speech_command_dataset` and extract the zip file inside this folder

- Run the command `pip install -r requirements.txt` to download all the python dependencies

- Run the `break_background_wav_to_1_sec.py` to break the all the wav files into 1 sec long wav files.( Only run this file once, Do not rerun if the wav files are already broken down into 1 sec long clips )

- Run the `processing_and_training.py` file to do the initial processing and training

- It will ask for the wake word you want to train on, you can input the wake word here ( Make sure it is the same as one of the directories present in the speech command dataset, if your preffered wake word is not present then follow instructions given below to add word directory to the speech_command_dataset )

## Add word directory in the speech_command_dataset

- Create a directory inside the `speech_command_dataset` with the name of preffered wake word

- Collect audio samples of multiple people saying the word, the audio format should be wav file.

- Store the wav files with the following specifications:

   | __Specifications__ | Value  |
   | ------------- | ------------- |
   | __Sample Rate__ | 16 KHz  |
   | __Number of Channels__ | 1 |
   | __Encoding__ | Signed 16-bit PCM |

- After completing these steps you can train the model using the same steps given above


## Instructions to get excel sheet of predictions made by the model

- Create a new folder with the name `test-dataset` and make two more folders inside it with the name `positive` and `negative`

- Place the wav files containing the wake word in the positive folder and all the other files in negative ( Taking input in different folder to print the actual ans in the excel sheet )

### Run Prediction with Docker

- Build the docker image using the command 
```
docker build -t csv_prediction -f Dockerfile.prediction_csv .
```
- To run the docker container write the following command
```
 sudo docker run -it -v "$(pwd)":/app csv_prediction
```
- The output will be a csv file with three colums, the colums are index, actual_ans, predicted_ans

### Run prediction without Docker

- If you have not already installed all the python dependencies then run `pip install -r requirements.txt`

- Run the `prediction_csv.py`

- The output will be a csv file with three colums, the colums are index, actual_ans, predicted_ans

## Run prediction on live audio stream

### Run live audio classification with Docker

- Build the docker image using the command 
```
docker build -t live_stream_classification -f Dockerfile.live_classification .
```

- To run the container use the following command
```
docker run -it -v "$(pwd)":/app --device /dev/snd:/dev/snd live_stream_classification
```
  __Note :__ The `--device /dev/snd:/dev/snd` is used to connect the host microphone with the docker container 

### Run live audio classification without Docker

- If you have not already installed all the python dependencies then run `pip install -r requirements.txt`

- Run the `live_audio_stream.py` file <br> <br>
  __Note :__ Some operating systems don't allow to take microphone input at 8000Khz, in that case try to run it with docker.

