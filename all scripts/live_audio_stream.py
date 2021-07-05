import sounddevice as sd
import numpy as np
import python_speech_features
from tensorflow.keras import models

word_threshold = 0.5
rec_duration = 0.5
window_stride = 0.5
sample_rate = 8000
num_channels = 1
num_mfcc = 16
model_path = 'wake_word_marvin_model_with_background_trained_on_all.h5'

window = np.zeros(int(rec_duration * sample_rate) * 2)

model = models.load_model(model_path)

def sd_callback(rec, frames, time, status):

    if status:
        print('Error:', status)

    rec = np.squeeze(rec)


    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec

    mfccs = python_speech_features.base.mfcc(window,
                                        samplerate=sample_rate,
                                        winlen=0.256,
                                        winstep=0.050,
                                        numcep=num_mfcc,
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

    val = model.predict(np.expand_dims(x_test,0))


    if val > word_threshold:
        print('activated')
    else:
        print(val[0][0])

with sd.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback):
    while True:
        pass
