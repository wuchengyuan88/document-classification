#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Reference: https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
import time
start_time = time.time()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import re
#import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding,SpatialDropout1D,LSTM,Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split

tf.random.set_seed(1)
np.random.seed(1)


df = pd.read_csv('../xydata.csv')

# Data Summary
df.info()
print(df.Y.value_counts())

# Data visualization
plot1 = plt.figure(1)
seaborn.countplot(x='Y', data=df,color='blue')
plt.savefig('histogram.png')


# Text preprocessing
df = df.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    

    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwords from text
    return text
df['X'] = df['X'].apply(clean_text)
df['X'] = df['X'].str.replace('\d+','')

# df['X'][0]

################ LSTM Modeling
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each document.
#print(df.X.map(lambda x: len(x)).max())
# 3244
#print(df.X.map(lambda x: len(x)).min())
# 55
#print(df.X.map(lambda x: len(x)).mean())
# 1064.7346582984658
MAX_SEQUENCE_LENGTH = 500
print(r'MAX_SEQUENCE_LENGTH: '+str(MAX_SEQUENCE_LENGTH))
# This is fixed.
EMBEDDING_DIM = 100
# word2vec uses dimension of 300

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['X'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
#Found 46660 unique tokens.

# Truncate and pad the input sequences so that they are 
# all in the same length for modeling.
X = tokenizer.texts_to_sequences(df['X'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)
# Shape of data tensor: (5736, 500)

# Create label tensor
Y = pd.get_dummies(df['Y']).values
print('Shape of label tensor:', Y.shape)
# Shape of label tensor: (5736, 11)

# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1,
                                                    random_state = 1)
print('X_train.shape, Y_train.shape:')
print(X_train.shape,Y_train.shape)
# (5162, 250) (5162, 11)
print('X_test.shape, Y_test.shape:')
print(X_test.shape,Y_test.shape)
# (574, 250) (574, 11)

# Preparing the Embedding layer
# We compute an index mapping words to known embeddings, by parsing the data dump of pre-trained embeddings:
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
embeddings_index = {}
f = open('../../word2vec_data/modelutf8.txt')
#head = [next(f) for x in range(10)]
#print(head)

for line in f:
    values = line.split()
    word = values[0]
    #print(word)
    coefs = np.asarray(values[1:], dtype='float32')
    #print(coefs)
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
# Found 4027170 word vectors.

# we can leverage our embedding_index dictionary and our word_index 
# to compute our embedding matrix:
embedding_matrix = np.zeros((MAX_NB_WORDS, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# Model
# We load this embedding matrix into an Embedding layer. 
# Note that we set trainable=False to prevent the weights from being updated during training.
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM,weights=[embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH,trainable=False))
model.add(SpatialDropout1D(0.2))
#model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
#model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(11, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(), 
              metrics=['accuracy'])

epochs = 100
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                    validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', 
                                             mode='min', patience=10)])

# Evaluate model accuracy
accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

# Plot Loss
plot2 = plt.figure(2)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.savefig('loss.png')


# Plot Accuracy
plot3 = plt.figure(3)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.legend()
plt.savefig('accuracy.png')
plt.show()

################
print("--- %s seconds ---" % (time.time() - start_time))
