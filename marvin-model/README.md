# Instructions to run the wake word model trained on the word "Marvin" on a raspberry pi

- Put both the files `04-rpi-tflite-audio-stream.py` and the model `wake_word_marvin_lite.tflite` into a Raspberry Pi.

- Connect a resistor and LED to board pin 8 in the Raspberry Pi.

- Run the script `04-rpi-tflite-audio-stream.py`

- Whenever you say "Marvin", the LED should flash briefly.
