import numpy as np
import pandas as pd
import torch
import time
import json
from tqdm import tqdm
from torch_geometric.data import Data
import argparse

from models.Model import MKGRec
from utils import MyLoader, init_seed, generate_kg_batch, add_random_edges, drop_random_edges
from evaluate import eval_model 
from prettytable import PrettyTable
from torch_geometric.utils import coalesce

def parse_args():
    parser = argparse.ArgumentParser(description="Run MKGRec.")
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed.')
    parser.add_argument('--dataset', nargs='?', default='dbbook2014',
                        help='Choose a dataset from {yelp2018, last-fm, amazon-book}')
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    args = parser.parse_args()
    return args

args = parse_args()
dataset = args.dataset
config = json.load(open(f'./config/{dataset}.json')) 
config['device'] = f'cuda:{args.gpu_id}' 
seed = args.seed 
Ks = config['Ks'] 
init_seed(seed, True)

print('-'*100)
for k, v in config.items():
    print(f'{k}: {v}')
print('-'*100)

loader = MyLoader(config)
src = loader.train.loc[:, 'userid'].to_list()+ loader.train.loc[:, 'itemid'].to_list()
tgt = loader.train.loc[:, 'itemid'].to_list() + loader.train.loc[:, 'userid'].to_list()
src_cf = loader.train.loc[:, 'userid'].to_list()+ loader.train.loc[:, 'itemid'].to_list()
tgt_cf = loader.train.loc[:, 'itemid'].to_list() + loader.train.loc[:, 'userid'].to_list()
edge_index = [src, tgt]
src_k = loader.kg_org['head'].to_list() + loader.kg_org['tail'].to_list()
tgt_k = loader.kg_org['tail'].to_list() + loader.kg_org['head'].to_list()
edge_index[0].extend(src_k)
edge_index[1].extend(tgt_k)
def calculate_mutual_information(user_interest_df):
    users = user_interest_df['uid'].unique()
    interests = user_interest_df['interest'].unique()
    num_users = len(users)
    num_interests = len(interests)
    user_interest_matrix = np.zeros((num_users, num_interests))

    for index, row in user_interest_df.iterrows():
        user_index = np.where(users == row['uid'])[0][0]
        interest_index = np.where(interests == row['interest'])[0][0]
        user_interest_matrix[user_index, interest_index] = 1

    p_user = user_interest_matrix.sum(axis=1) / num_interests
    p_interest = user_interest_matrix.sum(axis=0) / num_users
    p_user_interest = user_interest_matrix / num_users

    mutual_information = np.zeros((num_users, num_interests))
    for i in range(num_users):
        for j in range(num_interests):
            if p_user_interest[i, j] > 0:
                mutual_information[i, j] = p_user_interest[i, j] * np.log(
                    p_user_interest[i, j] / (p_user[i] * p_interest[j]))

    max_mi = np.max(mutual_information)
    if max_mi > 0:
        mutual_information = mutual_information / max_mi

    return mutual_information

mutual_information = calculate_mutual_information(loader.kg_interest)

src_in = loader.kg_interest['uid'].to_list() + loader.kg_interest['interest'].to_list()
tgt_in = loader.kg_interest['interest'].to_list() + loader.kg_interest['uid'].to_list()
users = loader.kg_interest['uid'].unique()
interests = loader.kg_interest['interest'].unique()
edge_weights_ig = []
for i in range(len(src_in)):
    if src_in[i] in users and tgt_in[i] in interests:
        user_index = np.where(users == src_in[i])[0][0]
        interest_index = np.where(interests == tgt_in[i])[0][0]
        weight = mutual_information[user_index, interest_index]
        edge_weights_ig.append(weight)
    elif src_in[i] in interests and tgt_in[i] in users:
        user_index = np.where(users == tgt_in[i])[0][0]
        interest_index = np.where(interests == src_in[i])[0][0]
        weight = mutual_information[user_index, interest_index]
        edge_weights_ig.append(weight)
    else:
        edge_weights_ig.append(0)

if len(edge_weights_ig) != len(src_in):
    print(f"Error: Edge weights length {len(edge_weights_ig)} does not match edge index length {len(src_in)}.")
    raise ValueError("Edge weights and edge index lengths must be the same.")

for _ in range(len(src_cf)):
    edge_weights_ig.append(0)

edge_index_ig = [src_in + src_cf, tgt_in + src_cf]
edge_index_ig = torch.LongTensor(edge_index_ig)
edge_weights_ig = torch.FloatTensor(edge_weights_ig)
edge_index_ig, edge_weights_ig = coalesce(edge_index_ig, edge_weights_ig)
graph_ig = Data(edge_index=edge_index_ig.contiguous(), edge_attr=edge_weights_ig)
graph_ig = graph_ig.to(config['device'])
print(f'Is ig no duplicate edge: {graph_ig.is_coalesced()}')

edge_index_kg = [src_k+src_cf, tgt_k+src_cf]
edge_index_kg = torch.LongTensor(edge_index_kg)
edge_index_kg = coalesce(edge_index_kg)
graph_kg = Data(edge_index=edge_index_kg.contiguous())
graph_kg = graph_kg.to(config['device'])
print(f'Is kg no duplicate edge: {graph_kg.is_coalesced()}')
edge_index = torch.LongTensor(edge_index)
edge_index = coalesce(edge_index)
graph = Data(edge_index=edge_index.contiguous())
graph = graph.to(config['device'])
print(f'Is cikg no duplicate edge: {graph.is_coalesced()}')
edge_index_cf = torch.LongTensor([src, tgt])
edge_index_cf = coalesce(edge_index_cf)
graph_cf = Data(edge_index=edge_index_cf.contiguous())
graph_cf = graph_cf.to(config['device'])
print(f'Is cg no duplicate edge: {graph_cf.is_coalesced()}')

model = MKGRec(config, graph.edge_index)
model = model.to(config['device'])
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

if config['use_kge']:
    kg_optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_kg'])

train_loader, _ = loader.get_cf_loader(bs=config['batch_size'])

best_res = None
best_loss = 1e10
best_score = 0
patience = config['patience']
best_performance = []

total_start_time = time.time()

for epoch in range(config['num_epoch']):
    loss_list = []
    print(f'epoch {epoch+1} start!')
    model.train()
    start = time.time() 

    model.gmae_p = model.calc_mask_rate(epoch)
    num_nodes = model.users + model.entities  
    
    for data in train_loader:
        user, pos, neg = data
        user, pos, neg = user.to(config['device']), pos.to(config['device']), neg.to(config['device'])
        optimizer.zero_grad()
        loss_rec = model(user.squeeze(), pos.squeeze(), neg.squeeze())
        
        if model.align_weight > 0:
            loss_cross_domain_contrastive, all_layer_cf, all_layer_interest, all_layer_kg, meta_view = model.cross_domain_contrastive_loss(
                user.squeeze(), pos.squeeze(), 
                graph_ig.edge_index, graph_kg.edge_index, graph_cf.edge_index,
                return_intermediate=True
            )
            
            if model.align_strategy == 'selective':
                meta_view_strong = (model.transfer_net_cf(all_layer_cf) * model.cf_weight + 
                                   model.transfer_net_kg(all_layer_kg)) / 2
                loss_align = model.meta_knowledge_alignment_loss_selective(
                    user.squeeze(), pos.squeeze(),
                    all_layer_interest, meta_view_strong
                )
            else:
                loss_align = model.meta_knowledge_alignment_loss_v2(
                    user.squeeze(), pos.squeeze(),
                    all_layer_cf, all_layer_interest, all_layer_kg, meta_view
                )
            
            loss_interest_recon = model.interest_recon_loss(graph.edge_index)
            loss = loss_rec + loss_cross_domain_contrastive + loss_interest_recon + model.align_weight * loss_align
        else:
            loss_cross_domain_contrastive = model.cross_domain_contrastive_loss(
                user.squeeze(), pos.squeeze(), 
                graph_ig.edge_index, graph_kg.edge_index, graph_cf.edge_index
            )
            loss_interest_recon = model.interest_recon_loss(graph.edge_index)
            loss = loss_rec + loss_cross_domain_contrastive + loss_interest_recon
        
        loss.backward()
        optimizer.step()
        loss_list.append(loss.detach().cpu().numpy())
        
    sum_loss = np.sum(loss_list)/len(loss_list)
    if config['use_kge']:
        kg_total_loss = 0
        n_kg_batch = config['n_triplets'] // 4096
        for iter in range(1, n_kg_batch + 1):
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = generate_kg_batch(loader.kg_dict, 4096, config['entities']-config['interests'], 0)
            kg_batch_head = kg_batch_head.to(config['device'])
            kg_batch_relation = kg_batch_relation.to(config['device'])
            kg_batch_pos_tail = kg_batch_pos_tail.to(config['device'])
            kg_batch_neg_tail = kg_batch_neg_tail.to(config['device'])
    
            kg_batch_loss = model.get_kg_loss(kg_batch_head, kg_batch_pos_tail, kg_batch_neg_tail, kg_batch_relation)
    
            kg_optimizer.zero_grad()
            kg_batch_loss.backward()
            kg_optimizer.step()
            kg_total_loss += kg_batch_loss.item()
    
    if (epoch+1) % config['eval_interval'] == 0:            
        result = eval_model(model, loader, 'test')
        score = result['recall'][1] + result['ndcg'][1]
        if score > best_score:
            best_performance = []
            best_score = score
            patience = config['patience']
            for i, k in enumerate(Ks):
                table = PrettyTable([f'recall@{k}',f'ndcg@{k}',f'precision@{k}',f'hit_ratio@{k}'])
                table.add_row([round(result['recall'][i], 4),round(result['ndcg'][i], 4),round(result['precision'][i], 4), round(result['hit_ratio'][i], 4)])
                print(table)
                best_performance.append(table)
        else:
            patience -= 1
            print(f'patience: {patience}')

    if config['use_kge']:
        print(f'kg loss:{round(kg_total_loss / (n_kg_batch+1), 4)}')    
    print('epoch loss: ', round(sum_loss, 4))
    print('current mask rate:', round(model.gmae_p, 4))
    if patience <= 0:
        break
    end = time.time()
    print(f'train time {round(end-start)}s')
    print('-'*90)

total_end_time = time.time()
total_time = total_end_time - total_start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)

print('='*100)
print(f'Total Training Time: {hours}h {minutes}m {seconds}s ({round(total_time, 2)}s)')
print('='*100)

print('='*100)
print('Best testset performance:')
for table in best_performance:
    print(table)
