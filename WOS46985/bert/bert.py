#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# References: 
# https://huggingface.co/transformers/task_summary.html
# https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_tf_glue.py
# https://www.tensorflow.org/official_models/fine_tuning_bert
# https://stackabuse.com/text-classification-with-bert-tokenizer-and-tf-2-0-in-python/
# https://www.curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
# https://stackoverflow.com/questions/59978959/how-to-use-hugging-face-transformers-library-in-tensorflow-for-text-classificati
# https://www.kaggle.com/nkaenzig/bert-tensorflow-2-huggingface-transformers/
# https://stackoverflow.com/questions/61000500/tensorflow-keras-bert-multiclass-text-classification-accuracy
# https://stackoverflow.com/questions/60463829/training-tfbertforsequenceclassification-with-custom-x-and-y-data

import time
start_time = time.time()

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import re
#import nltk
from nltk.corpus import stopwords

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification, BertConfig
from transformers import TFBertModel
from transformers import create_optimizer

tf.random.set_seed(1)
np.random.seed(1)


# Reading the csv files
df = pd.read_csv('../xydata.csv')

# Data Summary
df.info()
print(df.Y.value_counts())

num_labels = len(df.Y.unique())
# 11

# Data visualization
plot1 = plt.figure(1)
seaborn.countplot(x='Y', data=df,color='blue')
plt.show()



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


#  BERT parameters
# Max length of encoded string(including special tokens such as [CLS] and [SEP]):
MAX_SEQUENCE_LENGTH = 128

# Standard BERT model with lowercase chars only:
PRETRAINED_MODEL_NAME = 'bert-base-uncased' 

# Batch size for fitting:
BATCH_SIZE = 64

# Number of epochs:
# The authors recommend only 2-4 epochs of training for fine-tuning BERT on 
# a specific NLP task
EPOCHS = 3

# Define the required model from pretrained BERT for sequence classification:
def create_model(max_sequence, model_name, num_labels):
    bert_model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # This is the input for the tokens themselves(words from the dataset after encoding):
    input_ids = tf.keras.layers.Input(shape=(max_sequence,), dtype=tf.int32, name='input_ids')

    # attention_mask - is a binary mask which tells BERT which tokens to attend and which not to attend.
    # Encoder will add the 0 tokens to the some sequence which smaller than MAX_SEQUENCE_LENGTH, 
    # and attention_mask, in this case, tells BERT where is the token from the original data and where is 0 pad token:
    attention_mask = tf.keras.layers.Input((max_sequence,), dtype=tf.int32, name='attention_mask')
    
    # Use previous inputs as BERT inputs:
    output = bert_model([input_ids, attention_mask])[0]

    # We can also add dropout as regularization technique:
    #output = tf.keras.layers.Dropout(rate=0.15)(output)

    # Provide number of classes to the final layer:
    output = tf.keras.layers.Dense(num_labels, activation='softmax')(output)

    # Final model:
    model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=output)
    return model

# Instantiate the model using defined function, and compile our model:
model = create_model(MAX_SEQUENCE_LENGTH, PRETRAINED_MODEL_NAME, df.Y.nunique())


# Low learning rate better for fine-tuning BERT
opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# Tokenization(converting text to tokens):
def batch_encode(X, tokenizer):
    return tokenizer.batch_encode_plus(
    X,
    max_length=MAX_SEQUENCE_LENGTH, # set the length of the sequences
    add_special_tokens=True, # add [CLS] and [SEP] tokens
    return_attention_mask=True,
    return_token_type_ids=False, # not needed for this type of ML task
    pad_to_max_length=True, # add 0 pad tokens to the sequences less than max_length
    return_tensors='tf'
)    
    
# Load the tokenizer:
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)    

# Create label tensor
Y = pd.get_dummies(df['Y']).values
print('Shape of label tensor:', Y.shape)
# Shape of label tensor: (5736, 11)


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    df.X.values, Y, test_size=0.1)


# Encode
X_train = batch_encode(X_train,tokenizer)
X_test = batch_encode(X_test,tokenizer)


history = model.fit(
    x=X_train.values(),
    y=y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping(monitor='val_loss', 
                                             mode='min', patience=10)]
)

# Evaluate model accuracy
accr = model.evaluate(X_test.values(),y_test)
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


# https://github.com/pytorch/pytorch/issues/5858
#if __name__ == '__main__':
#    run()
    
