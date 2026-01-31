import os
import time
import re
import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import clean_text

import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

dataset = 'book-crossing'
if dataset in ['dbbook2014', 'book-crossing']:
    field = 'books'
else:
    field = 'movies'
    
r = '_r'
C = 350
max_his_num = 30

cluster_type = 'tfidf'
data_path = f'./data/{dataset}'
output_path = f'batch_output/{dataset}_max{max_his_num}_output.jsonl'

if dataset in ['dbbook2014', 'ml1m']:
    item_list = pd.read_csv(data_path+'/item_list.txt', sep=' ')
    entity_list = pd.read_csv(data_path+'/entity_list.txt', sep=' ')
else:
    item_list = list()    
    entity_list = list()
    lines = open(data_path+'/item_list.txt', 'r').readlines()
    for l in lines[1:]:
        if l == "\n":
            continue
        tmps = l.replace("\n", "").strip()
        elems = tmps.split(' ')
        org_id = elems[0]
        remap_id = elems[1]
        if len(elems[2:]) == 0:
            continue
        title = ' '.join(elems[2:])    
        item_list.append([org_id, remap_id, title])
    item_list = pd.DataFrame(item_list, columns=['org_id', 'remap_id', 'entity_name'])
    lines = open(data_path+'/entity_list.txt', 'r').readlines()
    for l in lines[1:]:
        if l == "\n":
            continue
        tmps = l.replace("\n", "").strip()
        elems = tmps.split()
        org_id = elems[0]
        remap_id = elems[1]
        entity_list.append([org_id, remap_id])
    entity_list = pd.DataFrame(entity_list, columns=['org_id', 'remap_id'])

kg = pd.read_csv(data_path+'/kg_final.txt', sep=' ', names=['head', 'relation', 'tail'])
relation = pd.read_csv(data_path+'/relation_list.txt', sep=' ')
entity_list['remap_id'] = entity_list['remap_id'].astype(int)

relation_intent = kg['relation'].max() + 1
relation_sim = kg['relation'].max() + 2
items = item_list['remap_id'].to_list()

lower = 1
upper = 50
llm_answer = []
intent_triples = []
sim_triples = set()
intents_list = []
check_intent = []
waste = []

user_intents_dict = {}
with open(output_path, mode='r') as f:
    for answer in jsonlines.Reader(f):
        row_id = answer['custom_id']
        raw_intents = answer['content']
        clean_intents = [clean_text(it) for it in raw_intents.strip().split(',')]
        
        intents = []
        for it in clean_intents:
            if r == '':
                if len(it.split()) > lower and len(it.split()) <= upper:
                    intents.append(it)
                else:
                    print(f'waste:', it.split())
                    waste.append(it)
            elif r == '_r':
                if len(it) > lower and len(it) <= upper:
                    intents.append(it)
                else:
                    print(f'waste:', it)
                    waste.append(it)
        user_intents_dict[row_id] = intents

        for it in intents:
            intent_triples.append([row_id, it])
        intents_list.extend(intents)

def load_ratings(file_name):
    inter_mat = []
    lines = open(file_name, 'r').readlines()
    for l in lines:
        tmps = l.replace("\n", "").strip()
        inters = [int(i) for i in tmps.split(' ')]
        u_id, pos_ids = inters[0], inters[1:]
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    return pd.DataFrame(inter_mat, columns=['userid', 'itemid'])

def build_item2entity(item_list, dataset):
    if dataset in ['dbbook2014', 'ml1m']:
        item2entity = {row['remap_id']: str(row['entity_name']).split('/')[-1].replace('_', ' ')
                       for _, row in item_list.iterrows()}
    else:
        item2entity = {int(row['remap_id']): row['entity_name'] for _, row in item_list.iterrows()}
    return item2entity

ratings = load_ratings(data_path + '/train.txt')
item2entity = build_item2entity(item_list, dataset)

user_his_dict = {}
ratings['item_name'] = ratings['itemid'].map(item2entity)
valid_ratings = ratings[ratings['item_name'].notna()]
for uid, group in valid_ratings.groupby('userid'):
    user_his_dict[str(uid)] = group['item_name'].tolist()[-max_his_num:]

all_texts = []
all_labels = []
for uid in user_his_dict.keys():
    if uid in user_intents_dict:
        history_texts = user_his_dict[uid]
        intent_texts = user_intents_dict[uid]
        history_texts = [text for text in history_texts if text]
        intent_texts = [text for text in intent_texts if text]
        if history_texts and intent_texts:
            all_texts.extend(history_texts + intent_texts)
            all_labels.extend([0] * len(history_texts) + [1] * len(intent_texts))

if not all_texts or not all_labels:
    raise ValueError("No valid texts or labels for mutual information calculation.")

vectorizer = TfidfVectorizer(max_df=0.8, ngram_range=(1, 2))
features = vectorizer.fit_transform(all_texts)

if all(isinstance(label, int) for label in all_labels):
    labels = np.array(all_labels) 
else:
    le = LabelEncoder()
    labels = le.fit_transform(all_labels) 

print(features.shape[0])
chunk_size = 1500
num_chunks = (features.shape[0] + chunk_size - 1) // chunk_size

mi_scores = np.zeros(features.shape[1])
print(num_chunks)

for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, features.shape[0])
    chunk_features = features[start:end].toarray()
    chunk_labels = labels[start:end]
    chunk_mi_scores = mutual_info_classif(chunk_features, chunk_labels, discrete_features=False)
    mi_scores += chunk_mi_scores
    print(f"Calculated {mi_scores} chunks")

mi_scores /= num_chunks

mi_threshold = 0.01
selected_indices = [i for i, score in enumerate(mi_scores) if score > mi_threshold]
selected_features = features[:, selected_indices]

plt.hist(mi_scores, bins=50, alpha=0.75, color='blue')
plt.title('Distribution of Mutual Information Scores')
plt.xlabel('Mutual Information Score')
plt.ylabel('Number of Features')
plt.savefig('mi_scores_distribution.png')  
plt.close()

print(f'cluster num: {C}')
selected_features_dense = selected_features.toarray()  

kmeans = KMeans(n_clusters=C, n_init='auto') 
cluster_ids = kmeans.fit_predict(selected_features_dense)


output_dict = dict(zip(intents_list, cluster_ids))

entity_add = []
entity_max = entity_list['remap_id'].max()+1
for k, v in output_dict.items():
    entity_add.append([k, v+entity_max])
    output_dict[k] = v + entity_max
entity_add = pd.DataFrame(entity_add, columns=['org_id', 'remap_id'])

intent_triples_df = pd.DataFrame(intent_triples, columns=['uid', 'interest'])
intent_triples_df['merged_interest'] = intent_triples_df['interest'].map(output_dict)
user_num = len(intent_triples_df['uid'].unique())
print(f'user num:{user_num}')

intent_triples_df_group = intent_triples_df.groupby('merged_interest').count()
intent_triples_df_sort = intent_triples_df_group.sort_values(by='uid', ascending=False)

del_list = []
del_list.extend(list(intent_triples_df_sort[intent_triples_df_sort['uid']>=int(user_num/5)].index))
print(len(del_list))
del_list.extend(intent_triples_df_sort[intent_triples_df_sort['uid']==1].index)
intent_triples_df_filter = intent_triples_df[~intent_triples_df['merged_interest'].isin(del_list)]

print('interest graph edge num:', len(intent_triples_df_filter))
print('interest graph sparsity:', round(len(intent_triples_df_filter)/(user_num*C), 4)*100, '%')
kg_intent = intent_triples_df_filter.loc[:, ['uid', 'merged_interest']]
kg_intent.columns = ['uid', 'interest']
kg_intent.to_csv(data_path+f'/kg_interest_C{C}_{cluster_type}{r}.txt', sep=' ', index=False, header=None)
