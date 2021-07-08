from os import listdir
from os.path import isdir, join
import librosa
import random
import numpy as np
import python_speech_features
from tensorflow.keras import models
import pandas as pd
model_filename = 'wake_word_model.h5
dataset_path = 'test-dataset'
for name in listdir(dataset_path):
    if isdir(join(dataset_path, name)):
        print(name)

all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
print(all_targets)

num_samples = 0
for target in all_targets:
    print(len(listdir(join(dataset_path, target))))
    num_samples += len(listdir(join(dataset_path, target)))
print('Total samples:', num_samples)
target_list = all_targets
feature_sets_file = 'test_targets_mfcc_sets.npz'
perc_keep_samples = 1.0
sample_rate = 8000
num_mfcc = 16
len_mfcc = 16
filenames = []
y = []
for index, target in enumerate(target_list):
    print(join(dataset_path, target))
    filenames.append(listdir(join(dataset_path, target)))
    y.append(np.ones(len(filenames[index])) * index)
filenames = [item for sublist in filenames for item in sublist]
y = [item for sublist in y for item in sublist]
filenames_test = filenames
y_orig_test = y

def calc_mfcc(path):

    # Load wavefile
    signal, fs = librosa.load(path, sr=sample_rate)

    # Create MFCCs from sound clip
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

def extract_features(in_files, in_y):
    prob_cnt = 0
    out_x = []
    out_y = []

    for index, filename in enumerate(in_files):

        # Create path from given filename and target item
        path = join(dataset_path, target_list[int(in_y[index])],
                    filename)

        # Check to make sure we're reading a .wav file
        if not path.endswith('.wav'):
            continue

        # Create MFCCs
        mfccs = calc_mfcc(path)

        # Only keep MFCCs with given length
        if mfccs.shape[1] == len_mfcc:
            out_x.append(mfccs)
            out_y.append(in_y[index])
        else:
            print('Dropped:', index, mfccs.shape)
            prob_cnt += 1

    return out_x, out_y, prob_cnt

x_test, y_test, prob = extract_features(filenames_test,
                                          y_orig_test)

positive = 'positive'
positive_index = all_targets.index(positive)
y_test = np.equal(y_test,positive_index).astype('float64')
x_test = np.array(x_test)
x_test = x_test.reshape(x_test.shape[0],
                        x_test.shape[1],
                        x_test.shape[2],
                        1)

model = models.load_model(model_filename)

y_predictions = []
for i in range(len(x_test)):
    prediction_from_model = model.predict(np.expand_dims(x_test[i],0))
    if prediction_from_model > 0.35:
        prediction = 1
    else:
        prediction = 0
    y_predictions.append( prediction)

dict = {'index':[i for i in range(len(x_test))],'actual_ans':y_test, 'predicted_ans': y_predictions}
df = pd.DataFrame(dict)

df.to_csv('predictions.csv')
