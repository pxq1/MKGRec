import numpy as np
import pandas as pd
import torch
import time
import json
import os
from tqdm import tqdm
from torch_geometric.data import Data
import argparse

from models.Model import MKGRecec
from utils import MyLoader, init_seed, generate_kg_batch
from evaluate import eval_model
from prettytable import PrettyTable
from torch_geometric.utils import coalesce

def run_experiment(threshold, dataset, gpu_id, seed=2024):
    print(f'\n{"="*100}')
    print(f'Running experiment with denoising_threshold = {threshold}')
    print(f'{"="*100}\n')
    
    config = json.load(open(f'./config/{dataset}.json'))
    config['device'] = f'cuda:{gpu_id}'
    config['denoising_threshold'] = threshold
    
    Ks = config['Ks']
    init_seed(seed, True)

    loader = MyLoader(config)

    src = loader.train.loc[:, 'userid'].to_list() + loader.train.loc[:, 'itemid'].to_list()
    tgt = loader.train.loc[:, 'itemid'].to_list() + loader.train.loc[:, 'userid'].to_list()
    src_cf = loader.train.loc[:, 'userid'].to_list() + loader.train.loc[:, 'itemid'].to_list()
    tgt_cf = loader.train.loc[:, 'itemid'].to_list() + loader.train.loc[:, 'userid'].to_list()
    
    edge_index = [src, tgt]
    
    # KG edges
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
    
    # Interest graph
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
    
    for _ in range(len(src_cf)):
        edge_weights_ig.append(0)
    
    edge_index_ig = [src_in + src_cf, tgt_in + tgt_cf]
    edge_index_ig = torch.LongTensor(edge_index_ig)
    edge_weights_ig = torch.FloatTensor(edge_weights_ig)
    edge_index_ig, edge_weights_ig = coalesce(edge_index_ig, edge_weights_ig)
    graph_ig = Data(edge_index=edge_index_ig.contiguous(), edge_attr=edge_weights_ig)
    graph_ig = graph_ig.to(config['device'])
    
    # KG graph
    edge_index_kg = [src_k + src_cf, tgt_k + tgt_cf]
    edge_index_kg = torch.LongTensor(edge_index_kg)
    edge_index_kg = coalesce(edge_index_kg)
    graph_kg = Data(edge_index=edge_index_kg.contiguous())
    graph_kg = graph_kg.to(config['device'])
    
    # Full graph
    edge_index = torch.LongTensor(edge_index)
    edge_index = coalesce(edge_index)
    graph = Data(edge_index=edge_index.contiguous())
    graph = graph.to(config['device'])
    
    # CF graph
    edge_index_cf = torch.LongTensor([src, tgt])
    edge_index_cf = coalesce(edge_index_cf)
    graph_cf = Data(edge_index=edge_index_cf.contiguous())
    graph_cf = graph_cf.to(config['device'])
    
    model = MKGRecec(config, graph.edge_index)
    model = model.to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    if config['use_kge']:
        kg_optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_kg'])
    
    train_loader, _ = loader.get_cf_loader(bs=config['batch_size'])
    
    best_score = 0
    patience = config['patience']
    best_results = None
    
    for epoch in range(config['num_epoch']):
        loss_list = []
        model.train()
        start = time.time()
        
        model.gmae_p = model.calc_mask_rate(epoch)
        
        for data in train_loader:
            user, pos, neg = data
            user, pos, neg = user.to(config['device']), pos.to(config['device']), neg.to(config['device'])
            optimizer.zero_grad()
            loss_rec = model(user.squeeze(), pos.squeeze(), neg.squeeze())
            loss_cross_domain_contrastive = model.cross_domain_contrastive_loss(
                user.squeeze(), pos.squeeze(), 
                graph_ig.edge_index, graph_kg.edge_index, graph_cf.edge_index
            )
            loss_interest_recon = model.interest_recon_loss(graph.edge_index)
            loss = loss_rec + loss_cross_domain_contrastive + loss_interest_recon
            loss.backward()
            optimizer.step()
            loss_list.append(loss.detach().cpu().numpy())
        
        sum_loss = np.sum(loss_list) / len(loss_list)
        
        if config['use_kge']:
            kg_total_loss = 0
            n_kg_batch = config['n_triplets'] // 4096
            for iter in range(1, n_kg_batch + 1):
                kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = generate_kg_batch(
                    loader.kg_dict, 4096, config['entities'] - config['interests'], 0
                )
                kg_batch_head = kg_batch_head.to(config['device'])
                kg_batch_relation = kg_batch_relation.to(config['device'])
                kg_batch_pos_tail = kg_batch_pos_tail.to(config['device'])
                kg_batch_neg_tail = kg_batch_neg_tail.to(config['device'])
                
                kg_batch_loss = model.get_kg_loss(kg_batch_head, kg_batch_pos_tail, kg_batch_neg_tail, kg_batch_relation)
                
                kg_optimizer.zero_grad()
                kg_batch_loss.backward()
                kg_optimizer.step()
                kg_total_loss += kg_batch_loss.item()
        
        if (epoch + 1) % config['eval_interval'] == 0:
            result = eval_model(model, loader, 'test')
            score = result['recall'][1] + result['ndcg'][1]
            if score > best_score:
                best_score = score
                best_results = result
                patience = config['patience']
                print(f'Epoch {epoch+1}: New best score = {score:.4f}')
            else:
                patience -= 1
        
        if patience <= 0:
            print(f'Early stopping at epoch {epoch+1}')
            break
        
        end = time.time()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}: loss={sum_loss:.4f}, time={round(end-start)}s')
    
    return best_results, best_score

def main():
    parser = argparse.ArgumentParser(description="Sensitivity Analysis for Denoising Threshold")
    parser.add_argument('--dataset', type=str, default='dbbook2014',
                        help='Dataset name')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--thresholds', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9',
                        help='Comma-separated threshold values to test')
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(',')]
    
    print(f'Starting sensitivity analysis for denoising threshold')
    print(f'Dataset: {args.dataset}')
    print(f'Thresholds to test: {thresholds}')
    print(f'GPU: {args.gpu_id}')
    print(f'Seed: {args.seed}')
    
    all_results = []
    
    for threshold in thresholds:
        try:
            results, score = run_experiment(threshold, args.dataset, args.gpu_id, args.seed)
            results_serializable = {}
            for key, value in results.items():
                if isinstance(value, list):
                    results_serializable[key] = [float(v) if hasattr(v, 'item') else v for v in value]
                else:
                    results_serializable[key] = value
            
            all_results.append({
                'threshold': float(threshold),
                'score': float(score),
                'results': results_serializable
            })
            print(f'\nThreshold {threshold}: Best Score = {score:.4f}')
        except Exception as e:
            print(f'Error with threshold {threshold}: {str(e)}')
            continue
    
    output_dir = 'sensitivity_results'
    os.makedirs(output_dir, exist_ok=True)
    
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    all_results_serializable = convert_to_serializable(all_results)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = f'{output_dir}/threshold_sensitivity_{args.dataset}_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results_serializable, f, indent=4)
    
    print(f'\n{"="*100}')
    print('SENSITIVITY ANALYSIS SUMMARY')
    print(f'{"="*100}\n')
    
    config = json.load(open(f'./config/{args.dataset}.json'))
    Ks = config['Ks']
    
    summary_table = PrettyTable()
    summary_table.field_names = ['Threshold', 'Score'] + [f'Recall@{k}' for k in Ks] + [f'NDCG@{k}' for k in Ks]
    
    for res in all_results:
        row = [res['threshold'], f"{res['score']:.4f}"]
        for i in range(len(Ks)):
            row.append(f"{res['results']['recall'][i]:.4f}")
        for i in range(len(Ks)):
            row.append(f"{res['results']['ndcg'][i]:.4f}")
        summary_table.add_row(row)
    
    print(summary_table)
    
    csv_file = f'{output_dir}/threshold_sensitivity_{args.dataset}_{timestamp}.csv'
    df_data = []
    for res in all_results:
        row_data = {'threshold': res['threshold'], 'score': res['score']}
        for i, k in enumerate(Ks):
            row_data[f'recall@{k}'] = res['results']['recall'][i]
            row_data[f'ndcg@{k}'] = res['results']['ndcg'][i]
            row_data[f'precision@{k}'] = res['results']['precision'][i]
            row_data[f'hit_ratio@{k}'] = res['results']['hit_ratio'][i]
        df_data.append(row_data)
    
    df = pd.DataFrame(df_data)
    df.to_csv(csv_file, index=False)
    
    print(f'\nResults saved to:')
    print(f'  - {output_file}')
    print(f'  - {csv_file}')
    
    best_result = max(all_results, key=lambda x: x['score'])
    print(f'\nBest threshold: {best_result["threshold"]} with score {best_result["score"]:.4f}')

if __name__ == '__main__':
    main()
