import os
import pandas as pd
import numpy as np
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

if __name__ == '__main__':

    # Adding script arguments
    # Training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--max-len', type=int, default=256, metavar='N',
                        help='input max sequence length for training (default: 60)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--resume', type=bool, default=False, metavar='N',
                        help='Resume training from the latest checkpoint (default: False)')

    # Data parameter
    parser.add_argument('--train_file', type=str, default=None, metavar='N',
                        help='Training data file name')

    # SageMaker Parameters
    parser.add_argument('--sm-model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args = parser.parse_args()

    # Setting up path vars.
    train_file = args.train_file
    training_dir = args.data_dir
    chkpt_dir = r'/opt/ml/checkpoints/'

    clean_data_path = os.path.join(training_dir, train_file)
    df_cleaned = utils.load_pickle(clean_data_path)
    ckpt_path = r'/opt/ml/checkpoints/'


    # Loading the bert tokenizer and model
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Binary classificaiton
    num_classes = 2
    # Load or retrieve model
    # Restore from the latest checkpoint if specified.
    if args.resume:
        bert_model = TFBertForSequenceClassification.from_pretrained(ckpt_path)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)
        bert_model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
        print('Saved weights were loaded')
    else:
        print("Checkpoint was not loaded, creating new model.")
        bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=num_classes)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)

        bert_model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

    input_sentences = df_cleaned['pre_processed']
    labels = df_cleaned['label']
    print("Number of input sentences: {}\nNumber of Labels:{}".format(len(input_sentences), len(labels)))

    # Passing inputs to the BERT Tokenizer.
    input_ids = []
    attn_masks = []

    # Parameters should be fairly self-explanatory. We'll try with 256 length to
    # try to use most of the search output. May be computationally prohibitive.
    for sentence in input_sentences:
        bert_input = bert_tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=args.max_len, padding='max_length',
                                                truncation=True, return_attention_mask=True)
        input_ids.append(bert_input['input_ids'])
        attn_masks.append(bert_input['attention_mask'])
    input_ids = np.asarray(input_ids)
    attn_masks = np.asarray(attn_masks)
    labels = np.array(labels)
    print(len(input_ids), len(attn_masks), len(labels))


    print("Input shape: {}\nMask Shape: {}\nLabel Length: {}".format(input_ids.shape, attn_masks.shape, len(labels)))

    # Train/test split and check dimensions.
    x_train, x_val, y_train, y_val, train_mask, val_mask = train_test_split(input_ids, labels, attn_masks, test_size=0.2)
    print("Train dim check: {}\nVal dim check: {}".format(x_train.shape[0] == len(y_train) == train_mask.shape[0],
                                                          x_val.shape[0] ==len(y_val) ==val_mask.shape[0]))

    # Loss, metrics, optimizer, callbacks
    model_path = os.path.join(args.sm_model_dir, 'transformer')

    # callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,save_weights_only=True,monitor='val_loss', save_freq='epoch')]
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)]
    print('\nModel Summary',bert_model.summary())


    #Training time!
    history=bert_model.fit([x_train,train_mask],y_train,batch_size=args.batch_size,epochs=args.epochs,
                           validation_data=([x_val,val_mask],y_val), callbacks=callbacks)

    # Save the transformer model weights
    bert_model.save_pretrained(ckpt_path)

    # Save the model since script mode doesn't do this automatically. This isn't actually useful
    # because of the way transformer models are handled.
    bert_model.save(args.sm_model_dir + '/1')

    # Pickle the history.
    utils.save_pickle('/opt/ml/checkpoints/history.pickle', history.history)
