from os import listdir
from os.path import isdir, join
import librosa
import soundfile as sf
from tensorflow.io import gfile
import os

def make_multiple_1_sec_clips(path):
    sample, sr = librosa.load(path, sr = 16000)

    begin = 0
    index = 1
    while begin + 16000 <= len(sample):
        loc = path + str(index) + '_break.wav'
        index = index + 1
        new_wav = sample[begin: begin+16000]
        begin = begin + 16000
        sf.write(loc, new_wav, 16000, 'PCM_16')

    os.remove(path)

location_of_background_noise = '/home/rishabh/testing wake word/background_1_sec/_background_noise_'

background_files = gfile.glob(location_of_background_noise+'/*.wav')

for i in background_files:
#     print(i)
    make_multiple_1_sec_clips(i)
