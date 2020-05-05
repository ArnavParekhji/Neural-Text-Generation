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

def pad_punct(text):
    text = re.sub('([.,!?()])', r' \1 ', text)
    text = re.sub('\s{2,}', ' ', text)
    return text

def remove_punct(text):
    punctuations = '''()-[]{};:'"\<>/@#$%^&*_~'''
    no_punct = ""
    for char in text:
        if char not in punctuations:
            no_punct = no_punct + char
    final_str = re.sub(' +', ' ', no_punct)
    return final_str

def clean_doc(doc):
    doc = doc.replace('--', ' ')
    punct_str = ",.?!"
    # re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    re_punc = remove_punct(pad_punct(doc))
    tokens= re_punc.split()
    # tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha() or word in punct_str]
    tokens = [word.lower() for word in tokens]
    return tokens

def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

in_filename = "republic.txt"
doc = load_doc(in_filename)

tokens = clean_doc(doc)

print(tokens[:200])
print("Total tokens:", str(len(tokens)))
print("Unique tokens:", str(len(set(tokens))))

length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
    seq = tokens[i-length : i]
    line = " ".join(seq)
    sequences.append(line)

out_filename = 'republic_sequences.txt'
save_doc(sequences, out_filename)