"""
This file has been authored by : Rishabh Vasudevan github:rishabh-vasudevan

This file is licensed under APL-2(Apache 2)
"""
import sounddevice as sd
import numpy as np
import python_speech_features
from tensorflow.keras import models
from scipy.io.wavfile import write
import uuid


model_path = 'wake_word_model.h5'

model = models.load_model(model_path)

recording_window = np.zeros(8000)

print("Listening for wake word")

def record():
    print("activated\n")
    print("recording")
    recording = sd.rec(80000,
                   samplerate=16000, channels=2)
    sd.wait()

    unique_filename = str(uuid.uuid4())

    write(unique_filename + ".wav", 16000, recording)

    print("Recording saved\n")
    print("Listening for wake word")


def sd_callback(rec, frames, time, status):

    if status:
        print('Error:', status)

    rec = np.squeeze(rec)


    recording_window[:len(recording_window)//2] = recording_window[len(recording_window)//2:]
    recording_window[len(recording_window)//2:] = rec

    mfccs = python_speech_features.base.mfcc(recording_window,
                                        samplerate=8000,
                                        winlen=0.256,
                                        winstep=0.050,
                                        numcep=16,
                                        nfilt=26,
                                        nfft=2048,
                                        preemph=0.0,
                                        ceplifter=0,
                                        appendEnergy=False,
                                        winfunc=np.hanning)
    mfccs = mfccs.transpose()



    x_test = mfccs
    x_test = np.array(x_test)
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)


    pred = np.expand_dims(x_test,0)

    val = model(pred,training = False)


    if val > 0.95:
        record()


with sd.InputStream(channels=1,
                    samplerate=8000,
                    blocksize=4000,
                    callback=sd_callback):
    while True:
        pass
