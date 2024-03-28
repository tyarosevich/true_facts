from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
# from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import utils
import time
from importlib import reload

# load_dotenv()
pd.options.mode.chained_assignment = None  # default='warn'

# key 1 is mine, key 2 is other acct.
api_key = os.environ.get('api_key2')
cse_id = os.environ.get('custom_search_eng')

# The service object to access the api.
service_obj = build("customsearch", "v1", developerKey=api_key)

# Get the pre-processed statements dataset and change
# nan to '' because reasons.
df_truefalse=pd.read_csv('data/true_false_dataset.csv')
df_truefalse.fillna('', inplace=True)
# Indexes where the search failed to produce an
# acceptable dict.
failed_indexes = utils.load_pickle('data/failed_indexes.pickle')

# Get the index of the first unprocessed row.
results_list = list(df_truefalse['search_results'])
is_empty = [len(x) for x in results_list]
if is_empty[0] == 0:
    start_id = 0
else:
    start_id = np.nonzero(is_empty)[0][-1] + 1

#%%
# Query the next n statements and store the search results as a list
# of the search summaries.
m = 9800
n = start_id + m
for i in range(start_id, n):
    query = df_truefalse['claim'][i]
    query = query.replace('"', '')
    try:
        result = utils.google_search(service_obj, query, cse_id)
    except HttpError:
        failed_indexes.append(i)
        time.sleep(30)
    except KeyError:
        failed_indexes.append(i)
    try:
        snippets = [x['snippet'] for x in result['items']]
        df_truefalse.at[i, 'search_results'] = snippets
    except KeyError:
        failed_indexes.append(i)
    time.sleep(0.5)

df_truefalse.to_csv(path_or_buf='data/true_false_dataset.csv', index=False)
utils.save_pickle('data/failed_indexes.pickle', failed_indexes)
