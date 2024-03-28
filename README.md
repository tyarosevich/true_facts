# True Facts
## Leveraging google search summaries and the googlesearch API to evaluate factual statements

# (i) Project Overview
One of the things that makes machine learning so powerful is its ability to classify data sets based on very weak signalling of information contained therein. This is the somewhat 'magical' ability that ML has to identify associations where there seems to be only noisy data. One of the problems with using machine learning models to identify true statements is that both true and false statements can be structured in an identical manner, and the quality of being 'true' or 'false' is only verifiable with outside information. For example, if I say 'Sean Connery was born in England' you have no way of evaluating the truthfulness of this statement without having an extrinsic reference. In other words, from the point of view of language in and of itself, this statement is identical to the the truth. No matter how goodd our machine learning models are, this statment has no information, weakly signalled or othwerise, to indicate whether or not its true and thus the classifier has nothing to work with. 

Where was Sean Connery born? I correctly assumed Scotland, but I had to double-check on wikipedia, i.e. an extrinsic source. One can also simply google 'Where was Sean Connery born?' and they'll receive an accurate answer based on Google's relevance algorithms. Finding extrinsic information for claims like this is trivial for a human with today's web, but cumbersome if we want to verify statements automatically. Furthermore, google results sometimes give conflicting information or obscure results that can be time consuming for a human to evaluate. But something is almost always true - Google gives *some* kind of result that is related to the query by relevance. It occurred to me that poring through vaguely related text to find weakly signalled extrinsic information is precisely the sort of task neural nets are good at performing. Thus this project is simply an attempt to classify statements as true or false by using the statement itself, and a summary of google search results based on the statement, as the feature space to train a moddel.

# (ii) Model structure and Data source
The model used for this classifier is the famous BERT (Bi-directional Encoder Representations from Transformers) model, which was produced by Google and released as an open source tool to the public. BERT is a natural language model that uses some recent developments in machine learning research to create a model that is extremely good at finding and using the relationships in long strings of text, even if they are far apart. This is precisely the kind of power I needed for this model, since a small snippet of text in the google search summary might be the signal that relates to the query statement and verifies or discards it. For example, if we search "Sean Connery was Born in England," one of the search results might contain 'born in Scotland'. BERT can then learn to pay attention to the relationship between those two phrases, and then in the process of training the model, it would hopefully learn that dissimilarity between the two phrases 'born in ___' is related to the false classification value. 

The dataset for this project comes from FEVER (Fact Extraction and VERification) dataset, a large set of statements that were tagged as verfied/non-verified/unverifiable via human examination (*Thorne, James and Vlachos, Andreas and Christodoulopoulos, Christos and Mittal, Arpit*, 2018). Code for this project was adapted from [this](https://towardsdatascience.com/steps-to-start-training-your-custom-tensorflow-model-in-aws-sagemaker-ae9cf7a205b9) article and [this](https://towardsdatascience.com/bert-for-dummies-step-by-step-tutorial-fb90890ffe03) article.

# (iii) Test model locally


```python
import deploy_utils
import time
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
import nltk
import keras
import pickle
from keras.models import Model
import keras.backend as K
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import itertools
from keras.models import load_model
from sklearn.utils import shuffle
from transformers import BertTokenizer, TFBertModel, BertConfig
from transformers import TFBertForSequenceClassification
import argparse
import json
from importlib import reload
from googleapiclient.discovery import build
import os
from dotenv import load_dotenv

load_dotenv()
# import boto3
import utils

bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)
bert_model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

bert_model = TFBertForSequenceClassification.from_pretrained('data/models/transformer_model/')

# Test query
query = "The fastest growing population in the United States is Hispanic"

clean_query, err_code = deploy_utils.get_input(query)

if err_code:
    output = 'There was an error searching your claim, most likely due to monthly quotas with the Google Search API'
else:
    input_id, attn_mask = deploy_utils.bert_tokenize(bert_tokenizer, clean_query)

pred = bert_model([input_id, attn_mask])
logit_softmax = tf.nn.softmax(pred.logits).numpy()
print(logit_softmax)

```

# (iv) Code to deploy model for training on sagemaker

import os
import sagemaker
from sagemaker import get_execution_role
import boto3
import pandas as pd
import time
import pickle
import tensorflow as tf
from sagemaker.tensorflow import TensorFlow



## Create sagemaker session and role


```python
# Create a SageMaker session to work with
sagemaker_session = sagemaker.Session()
try:
    role = sagemaker.get_execution_role()
    region = sagemaker_session.boto_session.region_name

except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='trent_sage')['Role']['Arn']
    region = sagemaker_session.boto_session.region_name

print(role)
print(region)

```

## Set Paths


```python
data_folder_name='data'
train_filename = 'cleaned_data.pickle'
# Set the directories for our model output
trainedmodel_path = 'trained_model'
output_data_path = 'output_data'
# Set the name of the artifacts that our model generate (model not included) 
model_info_file = 'model_info.pth'
train_file = os.path.abspath(os.path.join(data_folder_name, train_filename))

```


```python
# Specify bucket name
bucket_name = 'zennsunni-ml-sagemaker'
# Set the training data folder in S3
training_folder = r'true_facts/train'
# Set the output folder in S3
output_folder = r'true_facts'
# Set the checkpoint in S3 folder for our model 
ckpt_folder = r'true_facts/ckpt'

training_data_uri = r's3://' + bucket_name + r'/' + training_folder
output_data_uri = r's3://' + bucket_name + r'/' + output_folder
ckpt_data_uri = r's3://' + bucket_name + r'/' + ckpt_folder
```

## Upload training data and file structure to S3 


```python
inputs = sagemaker_session.upload_data(train_file,
                              bucket=bucket_name, 
                              key_prefix=training_folder)

```

# (v) Creating and fitting the Sagemaker estimator


```python
# The instance type to use. This is the cheapest single GPU instance.
# instance_type='ml.m5.2xlarge'
instance_type = 'ml.p2.xlarge'
# instance_type='local'

```


```python
# Create the Tensorflow estimator using a Tensorflow 2.1 container


### examine parameters
estimator = TensorFlow(entry_point='train.py',
                       source_dir="train",
                       # requirements_file='requirements.txt',
                       role=role,
                       instance_count=1,
                       instance_type=instance_type,
                       framework_version='2.2.0',
                       py_version='py37',
                       output_path=output_data_uri,
                       code_location=output_data_uri,
                       base_job_name='tf-transformer',
                       script_mode= True,
                       #checkpoint_local_path = 'ckpt', #Use default value /opt/ml/checkpoint
                       checkpoint_s3_uri = ckpt_data_uri,
                       hyperparameters={
                        'epochs': 1,
                        'resume': True,
                        'train_file': 'cleaned_data.pickle',
                       })
```


```python
# Set the job name and show it
job_name = 'truefacts-{}'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime()))
print(job_name)
```

    truefacts-2021-03-06-18-26-19



```python

# Call the fit method to launch the training job
estimator.fit({'training':training_data_uri}, job_name = job_name)
```


```python

```
