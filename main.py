import tensorflow as tf
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras import Model
import re
from tqdm import tqdm
import numpy as np
from collections import Counter
import os
import pickle

#####################
#PREPROCESS DATA
#####################
def preprocess_data(filename):
    '''
    with open(filename) as csv_file:
        labels = []
        train = []

        allData = ""
        songs = []

        print("Filtering data phase 1...")
        for i in tqdm(csv_file):
            allData += i

        print("Filtering data phase 2...")
        start = False
        curr_song = ""
        for i in tqdm(allData):
            if start == True:
                if i == '"':
                    start = False
                    songs += [curr_song]
                else:
                    curr_song += i
            elif start == False and i == '"':
                start = True
                curr_song = ""

        print("Filtering data phase 3...")
        train = []
        for song in tqdm(songs):
            filtered = re.sub('[.?!W#@,]', '', song)
            filtered = re.sub(r'\n', ' ', filtered)
            filtered = filtered.lower()
            temp = filtered.split(" ")
            train += [a for a in temp]

        allWords = train
        del train
        print(allWords[:1000])
        del allData
        counter = Counter(allWords)
        most_occur = counter.most_common(400000)
        del counter

        print("Filtering data phase 4...")
        vocab = [set[0] for set in most_occur]
        word2idx = {word:idx for idx, word in enumerate(vocab)}
        idx2word = {idx:word for idx, word in enumerate(vocab)}

        with open('utils/word2idx.p', 'wb') as fp:
            pickle.dump(word2idx, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open('utils/idx2word.p', 'wb') as fp:
            pickle.dump(idx2word, fp, protocol=pickle.HIGHEST_PROTOCOL)

        del vocab
        del most_occur

        print("Filtering data phase 5...")
        data = []
        for i in tqdm(range(len(allWords))):
            try:
                data += [word2idx[allWords[i]]]
            except:
                data += [400001]

        del allWords
        del word2idx
        del idx2word

        with open('data/data.p', 'wb') as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    '''
    print("Loading data...")
    data = []
    with open('data/data.p', 'rb') as fp:
        data = pickle.load(fp)
    return data

#####################
#BUILD MODEL
#####################
def create_model(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64, batch_input_shape=[32, None]),
        tf.keras.layers.LSTM(256,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.LSTM(256,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


#####################
#LOSS FUNCTION
#####################
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


#####################
#FUNCTION CALLS
#####################
data = preprocess_data("./data/data.p")

model = create_model(400000)
model.compile(optimizer='adam', loss=loss)

train = data
target = [train[-1]] + train[:-1]
del target[0]
del train[len(train)-1]

train = train[:12800]
target = target[:12800]

model.fit(train, target, epochs=100)

print(model.predict([[train[0]]]))


idx2word = {}
with open('idx2word.p', 'rb') as fp:
    idx2word = pickle.load(fp)

string = ""
for i in range(100):
    if data[i] != 400001:
        string += idx2word[data[i]] + " "

print(string)

del idx2word
del data