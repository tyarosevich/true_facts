import utils
import time
import re
import tensorflow as tf
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
import json

#%%

bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)
bert_model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

#%%
checkpoint_filepath = r'data/overfit_checkpoints/'   # /ckpt-1.data-00000-of-00001
#%%
ckpt = tf.train.Checkpoint(transformer=bert_model,
                           optimizer=optimizer)
#%%

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_filepath, max_to_keep=1)
ckpt.restore(ckpt_manager.latest_checkpoint)

#%%
bert_model.save('data' + '/1')

#%%
history = utils.load_pickle('data/2_24_history.pickle')