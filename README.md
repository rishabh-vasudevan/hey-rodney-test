# Instructions to train the wake word model and run it

- Clone the repository

- Download the Speech Command dataset folder from : https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz 

- Make a new folder with the name `speech_command_dataset` and extract the zip file inside this folder

- Run the `break_background_wav_to_1_sec.py` to break the all the wav files into 1 sec long wav files.( Only run this file once, Do not rerun if the wav files are already broken down into 1 sec long clips )

- Set the wake word you want to train the model on, you can change it in `processing_and_training.py` by changing the value of the variable `wake_word` on line 62 to the preffered wake word ( Preset to the word "Mavin", if you want to set it to something that is not already in the speech_command_dataset then you will have to create a new folder and add multiple wav files of your preffered wake word )

- Run the `processing_and_training.py` file to do the initial processing and training


## Instructions to get excel sheet of predictions made by the model

- Create a new folder with the name `test-dataset` and make two more folders inside it with the name `positive` and `negative`

- Place the wav files containing the wake word in the positive folder and all the other files in negative ( Taking input in different folder to print the actual ans in the excel sheet )

- Run the `prediction_csv.py`

- The output will be a csv file with three colums, the colums are index, actual_ans, predicted_ans

## To run prediction on live audio stream

- Run the `live_audio_stream.py` file


## Train the model using Docker

- Clone the repo

- Build the docker image

```
docker build -t processing-and-training .
```
 
- Mount the folder which contains the `speech_command_dataset` folder to the `/app` folder in the docker contianer
- It will breakdown the background noises if it is not already broken down
- It will ask for the wake word you want to train on, you can input the wake word here ( Make sure it is the same as one of the directories present in the speech command dataset, if your preffered wake word is not present then you can make a new directory and add 1 sec wav files containing the wake word )

```
docker run -it -v "$(pwd)":/app processing-and-training
```
