from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pickle
import re
import numpy as np
import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords
import unicodedata
from transformers import BertTokenizer
import os
from dotenv import load_dotenv
load_dotenv()

# Calls the google search API.
def google_search(service_obj, search_term, cse_id, **kwargs):
    '''
    Accesses the google JSON search API
    Parameters
    ----------
    search_obj:
    search_term: str
    cse_id: str
        The google custom search engine to be used.
    kwargs

    Returns: dict
        A JSON type object.
    -------
    '''
    res = service_obj.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res

def load_pickle(path):
    '''
    Loads a file
    Parameters
    ----------
    path: str
        local or full path of file

    Returns
    -------
    '''
    with open(path, 'rb') as f:
        file = pickle.load(f)
    return file

def save_pickle(path, item):
    '''
    Pickles a file
    Parameters
    ----------
    path: str
    item: Any

    Returns
    -------

    '''
    with open(path, 'wb') as f:
        pickle.dump(item, f)



def bert_tokenize(tokenizer, sentence):

    # Parameters should be fairly self-explanatory. We'll try with 256 length to
    # try to use most of the search output. May be computationally prohibitive.

    bert_input = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=256, padding='max_length',
                                            truncation=True, return_attention_mask=True)
    input_id = bert_input['input_ids']
    attn_mask = bert_input['attention_mask']

    input_id = np.asarray(input_id).reshape(1, 256)
    attn_mask = np.asarray(attn_mask).reshape(1, 256)

    return input_id, attn_mask

def get_input(query):
    '''
    Searches and processes the user's claim using the google API client.
    Parameters
    ----------
    query: str
    The user's query.

    Returns
    -------

    '''
    # key 1 is mine, key 2 is other acct.
    api_key = os.environ.get('google_search_api_key')
    cse_id = os.environ.get('google_search_eng_id')

    # Build the service object to search and remove quotes (they muck up google searches).
    service_obj = build("customsearch", "v1", developerKey=api_key)
    query = query.replace('"', '')
    query = clean_text(query)
    try:
        result = google_search(service_obj, query, cse_id)
        err_code = 0
        output = [x['snippet'] for x in result['items']]
        output = query + ' ' + clean_text(output)
    except HttpError as e:
        output = 'There was an error resolving the search, most likely because of a monthly quota.'
        err_code = e

    return output, err_code

def clean_text(string):
    '''
    Cleans a single string to prep it for searching.
    Parameters
    ----------
    string: str
    The claim to be verified.

    Returns: str
    -------
    '''

    # Apply each regex. Couldn't use regex for \n for reasons
    # unknown. Wasn't a raw text issue or a double escape issue.
    sub_conditions = {  # r'\n': '',  # Remove \n, newline symbols.
        '[A-Z]+': lambda m: m.group(0).lower(),  # Lowercase all
        '[^0-9a-zA-Z\s-]+': '',  # Keep only alphanumerc and spaces
        '.,!@:;?': ''  # Remove all punctuation.
    }
    reg_list = [(re.compile(a), b) for a, b in sub_conditions.items()]
    string = r"{}".format(string)
    string = ''.join(c for c in unicodedata.normalize('NFD', string) if unicodedata.category(c) != 'Mn')
    string = string.replace(r'\n', '')
    string = remove_stopwords(string)
    for tup in reg_list:
        string = tup[0].sub(tup[1], string)

    return string

