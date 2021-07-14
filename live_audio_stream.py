"""
This file has been authored by : Rishabh Vasudevan github:rishabh-vasudevan

This file is licensed under APL-2(Apache 2)
"""

import sounddevice as sd
import numpy as np
import python_speech_features
from tensorflow.keras import models

model_path = '/app/wake_word_model.h5'

model = models.load_model(model_path)

recording_window = np.zeros(8000)

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
        print('activated')
    else:
        print(val[0][0])



with sd.InputStream(channels=1,
                    samplerate=8000,
                    blocksize=4000,
                    callback=sd_callback):
    while True:
        pass
