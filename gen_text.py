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

def gen_seq(model, tokenizer, seq_legth, seed_text, n_words):
    result = list()
    in_text = seed_text
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = pad_sequences([encoded], maxlen=seq_length, padding="pre")
        yhat = model.predict_classes(encoded, verbose=0)
        out_word = ""
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        in_text += " " + out_word
        result.append(out_word)
    return " ".join(result)

in_filename = "republic_sequences.txt"
doc = load_doc(in_filename)
lines = doc.split("\n")

seq_length = len(lines[0].split()) - 1

model = load_model("lang_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

seed_text = lines[random.randint(0, len(lines))]
print(seed_text + "\n\n")

generated = gen_seq(model, tokenizer, seq_length, seed_text, 50)
print(generated)