#  Instructions to run the wake word model trained on the word "Marvin" on a raspberry pi

- Download the Speech Command dataset folder from : https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz

- Extract the zip folder and place it in the same location as the clone.

- Run the `01-speech-commands-mfcc-extraction.ipynb` script in a Jupyter Notebook( This will convert all the data present in the  “speech_command_dataset” to its equivalent MFCC.)

- Make a new folder with the name `feature_sets_directory` and move the `all_targets_mfcc_sets.npz` file inside this folder.

- Run the `02-speech-commands-mfcc-classifier.ipynb` script in a Jupyter Notebook. (This will train the “Marvin” model ).

- Run the `03-tflite-model-converter.ipynb script` in a Jupyter Notebook. ( This will convert the h5 file to a tflite file ).


## Deploy the model

- Put both the files `04-rpi-tflite-audio-stream.py` and the model `wake_word_marvin_lite.tflite` into a Raspberry Pi.

- Connect a resistor and LED to board pin 8 in the Raspberry Pi.

- Run the script `04-rpi-tflite-audio-stream.py`

- Whenever you say "Marvin", the LED should flash briefly.
