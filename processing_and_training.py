"""
This file is based on
01-speech-commands-mfcc-extraction.ipynb and 02-speech-commands-mfcc-classifier.ipynb file available on the github page of Shawn Hymel (https://github.com/ShawnHymel/tflite-speech-recognition)

The code was licensed under Beerware (https://en.wikipedia.org/wiki/Beerware)

Algorithm modified and changes made:

1. overlap of backgroud sound to increase accuracy when there is background noise, the background noise is selected at ramdom and overlapped with the wav files at different volumes randomly selected.

2. Changed the training model and also added dropout and regularization.

3. Added a multiplicity factor to remove the negative bias of the data.( also mixed with different backgound noise will in a way increase the positive data as whenever the same wav file is repeated it gets overlapped with a ramdom background noise ).

4. Added part to convert the .h5 model to tflite.

5. Mixed both ipynb file to one py file to make it easier to run.

6. Removed parts of the script which were not necessary for processing and training.

Information:

This file processes the wav files and converts them into MFCC's and then those MFCC's are used to train the wake word model.

The model is stored in the form of an .h5 file.

It is also later stored in the form of .tflite file.


"""

from os import listdir
from os.path import isdir, join
import librosa
import random
import numpy as np
import python_speech_features
import tensorflow_io as tfio
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.io import gfile
import tensorflow as tf

dataset_path = 'speech_command_dataset'
multiply_number = 20 #number of times you want to multiply the occurances of the wake word ( this is there to remove the negative bias )


all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]

all_targets.remove('_background_noise_')

target_list = all_targets
feature_sets_file = 'mfcc_with_background.npz'
perc_keep_samples = 1.0 # amount of sample ( for eg. 0.4 will only use 40% of the sample therefore it will reduce the audio processing time)
val_ratio = 0.1
test_ratio = 0.1
sample_rate = 8000
num_mfcc = 16
len_mfcc = 16

tflite_model_filename = 'wake_word_model_tflite.tflite'
model_filename = 'wake_word_model.h5'
wake_word = input("### Enter the wake word you want to train on ( Make sure there is a folder with the same name in the speech_command_dataset folder )\n")
check = True

while check:
    if wake_word in all_targets:
        check = False
    else:
        wake_word = input("### The wake word that you entered is not present in speech_command_dataset, please re-enter a word that is present in the speech_command_dataset\n")


filenames = []
y = []
for index, target in enumerate(target_list):
    filenames.append(listdir(join(dataset_path, target)))
    y.append(np.ones(len(filenames[index])) * index)


filenames = [item for sublist in filenames for item in sublist]
y = [item for sublist in y for item in sublist]

filenames_y = list(zip(filenames, y))
random.shuffle(filenames_y)
filenames, y = zip(*filenames_y)

filenames = filenames[:int(len(filenames) * perc_keep_samples)]

val_set_size = int(len(filenames) * val_ratio)
test_set_size = int(len(filenames) * test_ratio)

filenames_val = filenames[:val_set_size]
filenames_test = filenames[val_set_size:(val_set_size + test_set_size)]
filenames_train = filenames[(val_set_size + test_set_size):]

y_orig_val = y[:val_set_size]
y_orig_test = y[val_set_size:(val_set_size + test_set_size)]
y_orig_train = y[(val_set_size + test_set_size):]

def calc_mfcc(signal,fs):

    mfccs = python_speech_features.base.mfcc(signal,
                                            samplerate=fs,
                                            winlen=0.256,
                                            winstep=0.050,
                                            numcep=num_mfcc,
                                            nfilt=26,
                                            nfft=2048,
                                            preemph=0.0,
                                            ceplifter=0,
                                            appendEnergy=False,
                                            winfunc=np.hanning)
    return mfccs.transpose()

def add_background(path):

    signal, fs = librosa.load(path, sr = sample_rate)
    background_volume = np.random.uniform(0,0.1)
    background_files = gfile.glob(dataset_path + '/' +'_background_noise_' + '/*.wav')
    background_file = np.random.choice(background_files)
    background_sound,fs = librosa.load(background_file, sr = sample_rate)



    if len(background_sound) == len_mfcc and len(sample) == len_mfcc:
        signal = signal + background_sound*background_volume
    return calc_mfcc(signal,fs)

def extract_features(in_files, in_y):
    prob_cnt = 0
    out_x = []
    out_y = []

    for index, filename in enumerate(in_files):

        path = join(dataset_path, target_list[int(in_y[index])],
                    filename)

        if not path.endswith('.wav'):
            continue
        if target_list[int(in_y[index])] == wake_word:
            repeat = multiply_number
        else:
            repeat = 1

        for i in range(repeat):
            mfccs = add_background(path)

            if mfccs.shape[1] == len_mfcc:
                out_x.append(mfccs)
                out_y.append(in_y[index])
            else:
                print('Dropped:', index, mfccs.shape)
                prob_cnt += 1

    return out_x, out_y, prob_cnt

x_train, y_train, prob = extract_features(filenames_train,
                                          y_orig_train)
print('Removed percentage:', prob / len(y_orig_train))
x_val, y_val, prob = extract_features(filenames_val, y_orig_val)
print('Removed percentage:', prob / len(y_orig_val))
x_test, y_test, prob = extract_features(filenames_test, y_orig_test)
print('Removed percentage:', prob / len(y_orig_test))

np.savez(feature_sets_file,
         x_train=x_train,
         y_train=y_train,
         x_val=x_val,
         y_val=y_val,
         x_test=x_test,
         y_test=y_test)

wake_word_index = all_targets.index(wake_word)
y_train = np.equal(y_train, wake_word_index).astype('float64')
y_val = np.equal(y_val, wake_word_index).astype('float64')
y_test = np.equal(y_test, wake_word_index).astype('float64')

x_train = np.array(x_train)
x_val = np.array(x_val)
x_test = np.array(x_test)

x_train = x_train.reshape(x_train.shape[0],
                          x_train.shape[1],
                          x_train.shape[2],
                          1)
x_val = x_val.reshape(x_val.shape[0],
                      x_val.shape[1],
                      x_val.shape[2],
                      1)
x_test = x_test.reshape(x_test.shape[0],
                        x_test.shape[1],
                        x_test.shape[2],
                        1)

sample_shape = x_test.shape[1:]

model = Sequential([
    Conv2D(32, 3,
           padding='same',
           activation='relu',
           kernel_regularizer=regularizers.l2(0.001),
           name='conv_layer1',
           input_shape=sample_shape),
    MaxPooling2D(name='max_pooling1', pool_size=(2,2)),
    Conv2D(32, 3,
           padding='same',
           activation='relu',
           kernel_regularizer=regularizers.l2(0.001),
           name='conv_layer2'),
    MaxPooling2D(name='max_pooling2', pool_size=(2,2)),
    Conv2D(32, 3,
           padding='same',
           activation='relu',
           kernel_regularizer=regularizers.l2(0.001),
           name='conv_layer3'),
    MaxPooling2D(name='max_pooling3', pool_size=(2,2)),
    Flatten(),
    Dropout(0.2),
    Dense(
        40,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        name='hidden_layer1'
    ),
    Dense(
        1,
        activation='sigmoid',
        kernel_regularizer=regularizers.l2(0.001),
        name='output'
    )
])

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

history = model.fit(x_train,
                    y_train,
                    epochs=50,
                    batch_size=100,
                    validation_data=(x_val, y_val))

models.save_model(model, model_filename)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(tflite_model_filename,'wb') as f:
    f.write(tflite_model)
