from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
load_dotenv()
pd.options.mode.chained_assignment = None  # default='warn'
import timeit
#%%
my_api_key = os.environ.get('api_key')
my_cse_id = os.environ.get('custom_search_eng')

def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res

#%%

result = google_search('Coffee', my_api_key, my_cse_id)

#%%
path_true_statements = 'data/train.jsonl'
df_true = pd.read_json(path_true_statements, lines=True)

#%%
test_search = df_true['claim'][1]

result = google_search(test_search, my_api_key, my_cse_id)

#%%

# Dropping unecessary columns.
df_true.drop(labels=['id', 'verifiable', 'evidence'], inplace=True, axis=1)

# Removing inconclusive labels, switch to 0/1  booleans.
df_truefalse = df_true[df_true.label != 'NOT ENOUGH INFO']
df_truefalse.loc[df_truefalse['label']=='SUPPORTS', 'label']=1
df_truefalse.loc[df_truefalse['label']=='REFUTES', 'label']=0
df_truefalse['search_results'] = ''
df_truefalse.reset_index(inplace=True)
df_truefalse.reindex()

#%%
df_truefalse.to_csv(path_or_buf='data/true_false_dataset.csv', index=False)

#%%
df_truefalse=pd.read_csv('data/true_false_dataset.csv')
df_truefalse.fillna('', inplace=True)


#%% Data wrangling

snippets = [x['snippet'] for x in result['items']]

#%% So something like this

# import df_truefalse done above

# get id of the last row that ws processed from df_completed
results_list = list(df_truefalse['search_results'])
is_empty = [len(x) for x in results_list]
if is_empty[0] == 0:
    id_last = 0
else:
    id_last = np.nonzero(is_empty)[0][-1] + 1
#%%
# make a set of the completed id's
# set_complete = set(df_completed['id'])

# use a generator to get the next row to process
    # Generator can confirm next row is not in the set of completed.

# Generator doesn't save a ton of memory, but it does let us avoid
# creating a sub-frame of n rows from the existing unprocessed frame
# and we may have to stop early because of API limits etc.
# def get_search_query(df, id_last, set, n):
#     for i in range(n):
#         row = df[id_last + i]
#         if row['id'] in set:
#             pass
#         else:
#             yield row


# query google
n = id_last + 10
for i in range(id_last, n):
    result = google_search(row['claim'], my_api_key, my_cse_id)
    snippets = [x['snippet'] for x in result['items']]
    df_truefalse.at[i, 'search_results'] = snippets
# parse the result to the snippets
# add to the completed dataframe
# save completed to csv and close connection.

#%%
