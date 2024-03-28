import os
import pandas as pd
import numpy as np
import utils
import time
import re
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
import nltk
import keras
import pickle
from keras.models import Model
import keras.backend as K
from sklearn.metrics import confusion_matrix,f1_score,classification_report
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import itertools
from keras.models import load_model
from sklearn.utils import shuffle
from transformers import BertTokenizer, TFBertModel, BertConfig
from transformers import TFBertForSequenceClassification
import argparse

#%%

# Setting up path vars.

try:
    df_cleaned = utils.load_pickle('data/cleaned_data.pickle')

except FileNotFoundError:

    # Import the data and do some basic text cleaning for the BERT tokenizer.
    df_truefalse=pd.read_csv('data/true_false_dataset.csv')
    df_truefalse.fillna('', inplace=True)

    df_cleaned = utils.preproc_searches(df_truefalse)

    utils.save_pickle('data/cleaned_data.pickle', df_cleaned)


#%% Loading the bert tokenizer and model

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# Binary classificaiton
num_classes = 2
# Sequence uncased model is most appropriate for this task.
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=num_classes)


input_sentences = df_cleaned['pre_processed']
labels = df_cleaned['label']
print("Number of input sentences: {}\nNumber of Labels:{}".format(len(input_sentences), len(labels)))

#%% Passing inputs to the BERT Tokenizer.

try:
    input_ids = utils.load_pickle('data/input_ids.pickle')
    attn_masks = utils.load_pickle('data/attn_masks.pickle')
    labels = utils.load_pickle('data/labels.pickle')
except FileNotFoundError:

    input_ids = []
    attn_masks = []
    # Parameters should be fairly self-explanatory. We'll try with 256 length to
    # try to use most of the search output. May be computationally prohibitive.
    for sentence in input_sentences:
        bert_input = bert_tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=256, padding='max_length',
                                                truncation=True, return_attention_mask=True)
        input_ids.append(bert_input['input_ids'])
        attn_masks.append(bert_input['attention_mask'])
    input_ids = np.asarray(input_ids)
    attn_masks = np.asarray(attn_masks)
    labels = np.array(labels)
    print(len(input_ids), len(attn_masks), len(labels))

    utils.save_pickle('data/input_ids.pickle', input_ids)
    utils.save_pickle('data/attn_masks.pickle', attn_masks)
    utils.save_pickle('data/labels.pickle', labels)

print("Input shape: {}\nMask Shape: {}\nLabel Length: {}".format(input_ids.shape, attn_masks.shape, len(labels)))
#%% Train/test split and check dimensions.
x_train, x_val, y_train, y_val, train_mask, val_mask = train_test_split(input_ids, labels, attn_masks, test_size=0.2)
print("Train dim check: {}\nVal dim check: {}".format(x_train.shape[0] == len(y_train) == train_mask.shape[0],
                                                      x_val.shape[0] ==len(y_val) ==val_mask.shape[0]))

#%% Loss, metrics, optimizer, callbacks

log_path = 'data/tensorboard/tb_bert'
model_path = 'data/models/bert_model.h5'

callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=model_path,save_weights_only=True,monitor='val_loss',
                                                mode='min',save_best_only=True),keras.callbacks.TensorBoard(log_dir=log_path)]

print('\nModel Summary',bert_model.summary())

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)

bert_model.compile(loss=loss,optimizer=optimizer,metrics=[metric])

#%% Training time!
history=bert_model.fit([x_train,train_mask],y_train,batch_size=32,epochs=1,
                       validation_data=([x_val,val_mask],y_val),callbacks=callbacks)