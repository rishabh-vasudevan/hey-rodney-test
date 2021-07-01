# Instructions to train the "Hey Rodney" wake word model and run it on a Raspberry Pi

- Download the Speech Command dataset folder from : https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz

- Extract the zip folder and place it in the same location as the clone ( The words in this dataset will act as negative sample ).

- Download all the samples of “Hey Rodney” from the `hey-rodney-dataset` and put all of the wav files in a folder.

  *You will have to make all the samples 1 sec long by cropping it, make sure no part of the word is cut out, this will have to be done manually using any software ( eg : Audacity )*

- Put this folder inside the `speech_command_dataset` folder with the name `hey-rodney`

- Run the `01-speech-commands-mfcc-extraction.ipynb` script in a Jupyter Notebook( This will convert all the data present in the  “speech_command_dataset” to its equivalent MFCC.)

- Make a new folder with the name `feature_sets_directory` and move the `all_targets_mfcc_sets.npz` file inside this folder.

- Run the `02-speech-commands-mfcc-classifier.ipynb` script in a Jupyter Notebook. (This will train the “Hey Rodney” model ).

- Run the `03-tflite-model-converter.ipynb script` in a Jupyter Notebook. ( This will convert the h5 file to a tflite file ).

- Copy the  `hey_rodney_lite.tflite model` and the `04-rpi-tflite-audio-stream.py` script into a raspberry pi and connect a resistor and LED to board pin 8 and run this script.
Whenever you say "Hey Rodney", the LED should flash briefly.
