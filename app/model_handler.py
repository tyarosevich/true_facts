#### Custom inference classes for sagemaker endpoint API

from transformers import BertTokenizer, TFBertModel, BertConfig, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np
import json
from sagemaker_inference import decoder, encoder
import os

class ModelHandler(object):

    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.error = None
        self._context = None
        self.initialized = False
        self.model = None
        self.tokenizer = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        # Set some flags.
        self._context = context
        self.initialized = True

        # Get the model directory path from the context.
        properties = context.system_properties
        # Contains the url parameter passed to the load request
        model_dir = properties.get("model_dir")

        # Build the tokenizer and model at initialization as recommended by AWS docs.
        bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

        # Get the path to the weights and load them.
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        weights_path = model_dir + '/1/variables/variables'
        bert_model.load_weights(weights_path).expect_partial()
        self.model = bert_model

        # Load the tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


    def input_fn(self, input_data, content_type):
        '''
        An input handler that performs BERT tokenization on the inference string.
        Parameters
        ----------
        input_data: str
        The input string.
        Returns: tuple
        The integer token ids, and the attention mask, both lists, i.e. (input_id, attn_mask).
        -------
        '''
        # Load
        byte_array = input_data[0]['body']
        print(type(byte_array))
        input_dict = json.loads(byte_array.decode())
        print(input_dict)
        print(type(input_dict))
        input_string = input_dict['query']
        # Tokenizes the input, with entirely preset parameters to match those with which the model was trained.
        bert_input = self.tokenizer.encode_plus(input_string, add_special_tokens=True, max_length=256, padding='max_length',
                                           truncation=True, return_attention_mask=True)
        input_id = bert_input['input_ids']
        attn_mask = bert_input['attention_mask']

        # Turn them into numpy arrays for input.
        input_id = np.asarray(input_id).reshape(1, 256)
        attn_mask = np.asarray(attn_mask).reshape(1, 256)

        return input_id, attn_mask

    def predict_fn(self, data):
        '''
        Return a prediction using the custom model.
        Parameters
        ----------
        data: tuple
        The (input_ids, attn_mask) tuple
        model: The BERT tf model.

        Returns
        -------

        '''
        input_ids = data[0]
        attn_mask = data[1]

        return self.model([input_ids, attn_mask])

    def output_fn(self, prediction):
        """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.

        Args:
            prediction: a prediction result from predict_fn
            accept: type which the output data needs to be serialized

        Returns: output data serialized
        """
        logit_softmax = tf.nn.softmax(prediction.logits).numpy()
        out_json = {'false': str(logit_softmax[0][0]), 'true':str(logit_softmax[0][1])}
        return [out_json]

    def handle(self, data, context):
        '''
        Pre-processes, performs inference, post-processes, and returns final output.
        Parameters
        ----------
        data: str
        Incoming request
        context: mms context

        Returns: serialized numpy array of softmaxed logits.
        -------
        '''

        model_input = self.input_fn(data, 'application/json')
        model_output = self.predict_fn(model_input)

        return self.output_fn(model_output)

_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)