import pickle
import re
import numpy as np
import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords
import unicodedata
from transformers import BertTokenizer


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


def clean_text(reg_list, string):
    # Apply each regex. Couldn't use regex for \n for reasons
    # unknown. Wasn't a raw text issue or a double escape issue.
    string = r"{}".format(string)
    string = ''.join(c for c in unicodedata.normalize('NFD', string) if unicodedata.category(c) != 'Mn')
    string = string.replace(r'\n', '')
    string = remove_stopwords(string)
    for tup in reg_list:
        string = tup[0].sub(tup[1], string)
    return string

def preproc_searches(df):
    sub_conditions = {  # r'\n': '',  # Remove \n, newline symbols.
        '[A-Z]+': lambda m: m.group(0).lower(),  # Lowercase all
        '[^0-9a-zA-Z\s-]+': '',  # Keep only alphanumerc and spaces
        '.,!@:;?': ''  # Remove all punctuation.
    }
    reg_list = [(re.compile(a), b) for a, b in sub_conditions.items()]

    # Get the last row that was processed.
    results_list = list(df['search_results'])
    is_empty = [len(x) for x in results_list]
    if is_empty[0] == 0:
        last_row = 0
    else:
        last_row = np.nonzero(is_empty)[0][-1]

    # Get rid of unsearched claims.
    df_cleaned = df[0: last_row]

    # Clean the results and claims, and concatenate them for input.
    df_cleaned['search_results'] = [clean_text(reg_list, x) for x in df_cleaned['search_results']]
    df_cleaned['claim'] = [clean_text(reg_list, x) for x in df_cleaned['claim']]
    df_cleaned['pre_processed'] = df_cleaned['claim'] + '' + df_cleaned['search_results']

    return df_cleaned

