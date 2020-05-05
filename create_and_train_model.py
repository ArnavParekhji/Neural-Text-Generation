import re
import string
import numpy as np
import pickle
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential, load_model

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def define_model(vocab_size, seq_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_legth))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(vocab_size, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model

in_filename = "republic_sequences.txt"
doc = load_doc(in_filename)
lines = doc.split("\n")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

vocab_size = len(tokenizer.word_index) + 1

max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding="pre")

sequences = np.array(sequences)
print(sequences)

X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_legth = X.shape[1]

model = define_model(vocab_size, seq_legth)
model.fit(X, y, batch_size=128, epochs=100, verbose=1)

model.save("lang_model.h5")
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))