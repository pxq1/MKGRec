import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import time
import random
from collections import defaultdict
from functools import partial
import math
from tqdm import tqdm
from torch_geometric.nn import GCNConv
from torch_geometric.nn import LGConv
from torch_geometric.utils import dropout_edge, add_self_loops 
from torch_geometric.utils import  coalesce

from .loss_func import _L2_loss_mean, BPRLoss, InfoNCELoss, sce_loss
from torch.nn import TripletMarginLoss
torch.autograd.set_detect_anomaly(True)

class MKGRec(torch.nn.Module):
    def __init__(self, config, edge_index):
        super(MKGRec, self).__init__()
        self.config = config
        self.users = config['users']
        self.items = config['items']
        self.entities = config['entities']
        self.relations = config['relations']
        self.interests = config['interests']
        self.layer = config['layer']
        self.emb_dim = config['dim']
        self.weight_decay = config['l2_reg']
        self.l2_reg_kge = config['l2_reg_kge']
        self.cf_weight = config['cf_weight']
        self.edge_drop = config['edge_dropout']
        self.message_drop_rate = config['message_dropout']
        self.message_drop = nn.Dropout(p=self.message_drop_rate)

        self.gcl_weight = config['gcl_weight']
        self.gcl_temp = config['gcl_temp']
        self.eps = config['eps']

        self.user_entity_emb = nn.Embedding(self.users+self.entities, self.emb_dim)
        nn.init.xavier_uniform_(self.user_entity_emb.weight)
        self.rel_emb = nn.Embedding(self.relations, self.emb_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        
        self.encoder_layer_emb_mask = config['encoder_layer_emb_mask']
        self.decoder_layer_emb_mask = config['decoder_layer_emb_mask']
        self._replace_rate = config['replace_rate']
        self._mask_token_rate = 1.0 - self._replace_rate
        self._drop_edge_rate = config['edge_mask_drop_edge_rate']
        self.interest_recon_weight = config['interest_recon_weight']
        self.criterion_emb_mask = self.setup_loss_fn(config['emb_mask_loss'], config['emb_mask_loss_alpha'])
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.emb_dim))
        
        self.align_weight = config.get('align_weight', 0.0)
        self.align_strategy = config.get('align_strategy', 'lightweight')
        
        if 'align_loss_type' in config and 'align_strategy' not in config:
            old_type = config.get('align_loss_type', 'cosine')
            if old_type == 'selective':
                self.align_strategy = 'selective'
            else:
                self.align_strategy = 'lightweight'
        self.encoder_to_decoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.emb_dim * 2, self.emb_dim),
            nn.LayerNorm(self.emb_dim)  
        )

        for layer in self.encoder_to_decoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        
        self.gmae_p = None
        self.total_epoch = config['total_epoch']
        self.increase_type = config['increase_type']
        self.min_mask_rate = config['min_mask_rate']
        self.max_mask_rate = config['max_mask_rate']
        self.denoising_threshold = config.get('denoising_threshold', 0.5)
        self.propagate = LGConv(normalize=True)
        
        if config['add_self_loops']:
            edge_index, _ = add_self_loops(edge_index)
        self.edge_index = edge_index                  
        self.bpr = BPRLoss()
        self.device = config['device']
        self.transfer_net_cf = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim)
        ).to(self.device)
        
        self.transfer_net_interest = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim)
        ).to(self.device)
        
        self.transfer_net_kg = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim)
        ).to(self.device)

    def setup_loss_fn(self, loss_fn, alpha_l=2):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def compute(self, x=None, edge_index=None, perturbed=False, mess_drop=False, layer_num=None):
        if x == None:
            emb = self.user_entity_emb.weight
        else:
            emb = x
        if self.message_drop_rate > 0.0 and mess_drop:
            emb = self.message_drop(emb)
        all_layer = [emb]
        if layer_num != None:
            layers = layer_num
        else:
            layers = self.layer
        for layer in range(layers):
            if edge_index == None:
                emb = self.propagate(emb, self.edge_index)
            else:
                emb = self.propagate(emb, edge_index)
            if perturbed:
                random_noise = torch.rand_like(emb).to(self.config['device'])
                emb = emb + torch.sign(emb) * F.normalize(random_noise, dim=-1) * self.eps
            all_layer.append(emb)
        all_layer = torch.stack(all_layer, dim=1)
        all_layer = torch.mean(all_layer, dim=1)
        return all_layer
        
    def forward(self, user_idx, pos_item, neg_item):
        if self.edge_drop > 0.0:
            use_edge, _ = dropout_edge(edge_index=self.edge_index, force_undirected=True, p=self.edge_drop, training=self.training)
            all_layer = self.compute(edge_index=use_edge, mess_drop=True)
        else:
            all_layer = self.compute(mess_drop=True) 
        user_emb = all_layer[user_idx]
        pos_emb = all_layer[pos_item]
        neg_emb = all_layer[neg_item]
        
        users_emb_ego = self.user_entity_emb(user_idx)
        pos_emb_ego = self.user_entity_emb(pos_item)
        neg_emb_ego = self.user_entity_emb(neg_item)

        pos_score = (user_emb * pos_emb).squeeze()
        neg_score = (user_emb * neg_emb).squeeze()
        cf_loss = self.bpr(torch.sum(pos_score, dim=-1), torch.sum(neg_score, dim=-1))
        
        reg_loss = (1/2)*(users_emb_ego.norm(p=2).pow(2)+pos_emb_ego.norm(p=2).pow(2)+neg_emb_ego.norm(p=2).pow(2))/float(len(user_idx))
        loss = self.cf_weight*cf_loss + reg_loss*self.weight_decay

        return loss
    
    def cross_domain_contrastive_loss(self, user_idx, pos_item, edge_index_interest, edge_index_kg, edge_index_cf, return_intermediate=False):
        
        all_layer_cf = self.compute(edge_index=edge_index_cf, perturbed=True, mess_drop=False)
        all_layer_interest = self.compute(edge_index=edge_index_interest, perturbed=False, mess_drop=False)
        all_layer_kg = self.compute(edge_index=edge_index_kg, perturbed=False, mess_drop=False)
        
        meta_view = (self.transfer_net_cf(all_layer_cf)*self.cf_weight + self.transfer_net_interest(all_layer_interest)*self.interest_recon_weight + self.transfer_net_kg(all_layer_kg)) / 3
        
        edge_index_cf, edge_index_interest, edge_index_kg = self.meta_knowledge_aware_denoising(meta_view, edge_index_cf, edge_index_interest, edge_index_kg, threshold=self.denoising_threshold)
        
        all_layer_cf = self.compute(edge_index=edge_index_cf, perturbed=True, mess_drop=False)
        all_layer_interest = self.compute(edge_index=edge_index_interest, perturbed=False, mess_drop=False)
        all_layer_kg = self.compute(edge_index=edge_index_kg, perturbed=False, mess_drop=False)
        
        meta_view_denoised = (self.transfer_net_cf(all_layer_cf)*self.cf_weight + self.transfer_net_interest(all_layer_interest)*self.interest_recon_weight + self.transfer_net_kg(all_layer_kg)) / 3
        
        all_layer_cf = F.normalize(all_layer_cf, dim=1)
        all_layer_interest = F.normalize(all_layer_interest, dim=1)
        all_layer_kg = F.normalize(all_layer_kg, dim=1)
        
        user_view_cf, item_view_cf = all_layer_cf[user_idx], all_layer_cf[pos_item]
        user_view_interest, item_view_interest = all_layer_interest[user_idx], all_layer_interest[pos_item]
        user_view_kg, item_view_kg = all_layer_kg[user_idx], all_layer_kg[pos_item]
        
        pos_score_cf = torch.sum(user_view_cf * item_view_cf, dim=-1)
        pos_score_interest = torch.sum(user_view_interest * item_view_interest, dim=-1)
        pos_score_kg = torch.sum(user_view_kg * item_view_kg, dim=-1)
        
        neg_score_cf = torch.matmul(user_view_cf, all_layer_cf.t())
        neg_score_interest = torch.matmul(user_view_interest, all_layer_interest.t())
        neg_score_kg = torch.matmul(user_view_kg, all_layer_kg.t())
        
        loss_cf = self.info_nce_loss(user_view_cf, item_view_cf, neg_score_cf)
        loss_interest = self.info_nce_loss(user_view_interest, item_view_interest, neg_score_interest)
        loss_kg = self.info_nce_loss(user_view_kg, item_view_kg, neg_score_kg)
        
        total_loss = loss_cf + loss_interest + loss_kg
        
        if return_intermediate:
            return self.gcl_weight * total_loss, all_layer_cf, all_layer_interest, all_layer_kg, meta_view_denoised
        
        return self.gcl_weight * total_loss 

    def meta_knowledge_aware_denoising(self, meta_view, edge_index_cf, edge_index_interest, edge_index_kg, threshold=0.5):
        if threshold < 0:
            return edge_index_cf, edge_index_interest, edge_index_kg
        
        meta_view_normalized = F.normalize(meta_view, p=2, dim=1)
        similarity_matrix = torch.mm(meta_view_normalized, meta_view_normalized.t())
        
        edge_weight_cf = similarity_matrix[edge_index_cf[0], edge_index_cf[1]]
        edge_weight_interest = similarity_matrix[edge_index_interest[0], edge_index_interest[1]]
        edge_weight_kg = similarity_matrix[edge_index_kg[0], edge_index_kg[1]]
        
        percentile_threshold_cf = torch.quantile(edge_weight_cf, threshold)
        percentile_threshold_interest = torch.quantile(edge_weight_interest, threshold)
        percentile_threshold_kg = torch.quantile(edge_weight_kg, threshold)
        
        mask_cf = edge_weight_cf > percentile_threshold_cf
        mask_interest = edge_weight_interest > percentile_threshold_interest
        mask_kg = edge_weight_kg > percentile_threshold_kg
        
        edge_index_cf = edge_index_cf[:, mask_cf]
        edge_index_interest = edge_index_interest[:, mask_interest]
        edge_index_kg = edge_index_kg[:, mask_kg]
        
        return edge_index_cf, edge_index_interest, edge_index_kg
    
    def info_nce_loss(self, view1, view2, neg_scores):
        pos_score = torch.sum(view1 * view2, dim=-1)
        logits = torch.cat([pos_score.unsqueeze(1), neg_scores], dim=1)
        logits /= self.gcl_temp
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def meta_knowledge_alignment_loss(self, user_idx, pos_item, edge_index_cf, edge_index_interest, edge_index_kg):
        all_layer_cf = self.compute(edge_index=edge_index_cf, perturbed=False, mess_drop=False)
        all_layer_interest = self.compute(edge_index=edge_index_interest, perturbed=False, mess_drop=False)
        all_layer_kg = self.compute(edge_index=edge_index_kg, perturbed=False, mess_drop=False)
        
        meta_view = (
            self.transfer_net_cf(all_layer_cf) * self.cf_weight + 
            self.transfer_net_interest(all_layer_interest) * self.interest_recon_weight + 
            self.transfer_net_kg(all_layer_kg)
        ) / 3
        
        meta_view_sg = meta_view.detach()
        
        h_u_cf = all_layer_cf[user_idx]
        h_i_cf = all_layer_cf[pos_item]
        h_u_interest = all_layer_interest[user_idx]
        h_i_interest = all_layer_interest[pos_item]
        h_u_kg = all_layer_kg[user_idx]
        h_i_kg = all_layer_kg[pos_item]
        m_u = meta_view_sg[user_idx]
        m_i = meta_view_sg[pos_item]
        
        loss_cf_user = F.mse_loss(h_u_cf, m_u)
        loss_cf_item = F.mse_loss(h_i_cf, m_i)
        loss_interest_user = F.mse_loss(h_u_interest, m_u)
        loss_interest_item = F.mse_loss(h_i_interest, m_i)
        loss_kg_user = F.mse_loss(h_u_kg, m_u)
        loss_kg_item = F.mse_loss(h_i_kg, m_i)
        
        loss_align = (1/3) * (
            (loss_cf_user + loss_cf_item) +
            (loss_interest_user + loss_interest_item) +
            (loss_kg_user + loss_kg_item)
        )
        
        return loss_align
    
    def meta_knowledge_alignment_loss_cosine(self, user_idx, pos_item, edge_index_cf, edge_index_interest, edge_index_kg):
        all_layer_cf = self.compute(edge_index=edge_index_cf, perturbed=False, mess_drop=False)
        all_layer_interest = self.compute(edge_index=edge_index_interest, perturbed=False, mess_drop=False)
        all_layer_kg = self.compute(edge_index=edge_index_kg, perturbed=False, mess_drop=False)
        
        meta_view = (
            self.transfer_net_cf(all_layer_cf) * self.cf_weight + 
            self.transfer_net_interest(all_layer_interest) * self.interest_recon_weight + 
            self.transfer_net_kg(all_layer_kg)
        ) / 3
        
        meta_view_sg = meta_view.detach()
        
        all_layer_cf_norm = F.normalize(all_layer_cf, p=2, dim=1)
        all_layer_interest_norm = F.normalize(all_layer_interest, p=2, dim=1)
        all_layer_kg_norm = F.normalize(all_layer_kg, p=2, dim=1)
        meta_view_sg_norm = F.normalize(meta_view_sg, p=2, dim=1)
        
        h_u_cf = all_layer_cf_norm[user_idx]
        h_i_cf = all_layer_cf_norm[pos_item]
        h_u_interest = all_layer_interest_norm[user_idx]
        h_i_interest = all_layer_interest_norm[pos_item]
        h_u_kg = all_layer_kg_norm[user_idx]
        h_i_kg = all_layer_kg_norm[pos_item]
        m_u = meta_view_sg_norm[user_idx]
        m_i = meta_view_sg_norm[pos_item]
        
        loss_cf_user = 1 - F.cosine_similarity(h_u_cf, m_u, dim=1).mean()
        loss_cf_item = 1 - F.cosine_similarity(h_i_cf, m_i, dim=1).mean()
        loss_interest_user = 1 - F.cosine_similarity(h_u_interest, m_u, dim=1).mean()
        loss_interest_item = 1 - F.cosine_similarity(h_i_interest, m_i, dim=1).mean()
        loss_kg_user = 1 - F.cosine_similarity(h_u_kg, m_u, dim=1).mean()
        loss_kg_item = 1 - F.cosine_similarity(h_i_kg, m_i, dim=1).mean()
        
        loss_align = (1/3) * (
            (loss_cf_user + loss_cf_item) +
            (loss_interest_user + loss_interest_item) +
            (loss_kg_user + loss_kg_item)
        )
        
        return loss_align
    def meta_knowledge_alignment_loss_v2(self, user_idx, pos_item, 
                                         all_layer_cf, all_layer_interest, all_layer_kg,
                                         meta_view):
        meta_view_sg = meta_view.detach()
        
        h_u_cf = all_layer_cf[user_idx]
        h_i_cf = all_layer_cf[pos_item]
        h_u_interest = all_layer_interest[user_idx]
        h_i_interest = all_layer_interest[pos_item]
        h_u_kg = all_layer_kg[user_idx]
        h_i_kg = all_layer_kg[pos_item]
        m_u = meta_view_sg[user_idx]
        m_i = meta_view_sg[pos_item]
        
        h_u_cf = F.normalize(h_u_cf, p=2, dim=1)
        h_i_cf = F.normalize(h_i_cf, p=2, dim=1)
        h_u_interest = F.normalize(h_u_interest, p=2, dim=1)
        h_i_interest = F.normalize(h_i_interest, p=2, dim=1)
        h_u_kg = F.normalize(h_u_kg, p=2, dim=1)
        h_i_kg = F.normalize(h_i_kg, p=2, dim=1)
        m_u = F.normalize(m_u, p=2, dim=1)
        m_i = F.normalize(m_i, p=2, dim=1)
        
        loss_cf = -(F.cosine_similarity(h_u_cf, m_u, dim=1).mean() + 
                    F.cosine_similarity(h_i_cf, m_i, dim=1).mean()) / 2
        
        loss_interest = -(F.cosine_similarity(h_u_interest, m_u, dim=1).mean() + 
                         F.cosine_similarity(h_i_interest, m_i, dim=1).mean()) / 2
        
        loss_kg = -(F.cosine_similarity(h_u_kg, m_u, dim=1).mean() + 
                   F.cosine_similarity(h_i_kg, m_i, dim=1).mean()) / 2
        
        loss_align = (loss_cf + loss_interest + loss_kg) / 3
        
        return loss_align
    
    def meta_knowledge_alignment_loss_selective(self, user_idx, pos_item,
                                                 all_layer_interest, meta_view_strong):
        meta_view_sg = meta_view_strong.detach()
        
        h_u_interest = F.normalize(all_layer_interest[user_idx], p=2, dim=1)
        h_i_interest = F.normalize(all_layer_interest[pos_item], p=2, dim=1)
        m_u = F.normalize(meta_view_sg[user_idx], p=2, dim=1)
        m_i = F.normalize(meta_view_sg[pos_item], p=2, dim=1)
        
        loss_align = -(F.cosine_similarity(h_u_interest, m_u, dim=1).mean() + 
                      F.cosine_similarity(h_i_interest, m_i, dim=1).mean()) / 2
        
        return loss_align

    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes

        out_x[token_nodes] = 0.0
        out_x[token_nodes] += self.enc_mask_token

        return out_x, (mask_nodes, keep_nodes)

    def interest_recon_loss(self, edge_index):
        x = self.user_entity_emb.weight
        user_emb, entity_emb, interest_emb = torch.split(x, [self.users, self.entities-self.interests, self.interests])
        use_interest, (mask_nodes, keep_nodes) = self.encoding_mask_noise(interest_emb, self.gmae_p)
        mask_nodes = mask_nodes + (self.users + self.entities-self.interests)
        keep_nodes = keep_nodes + (self.users + self.entities-self.interests)
        use_x = torch.cat((user_emb, entity_emb, use_interest))

        if self._drop_edge_rate > 0.0:
            use_edge_index, masked_edges = dropout_edge(edge_index, force_undirected=True, p=self._drop_edge_rate, training=self.training)
            use_edge_index = add_self_loops(use_edge_index)[0]
        else:
            use_edge_index = edge_index

        enc_rep = self.compute(x=use_x, edge_index=use_edge_index, layer_num=self.encoder_layer_emb_mask)
        rep = self.encoder_to_decoder(enc_rep) + enc_rep

        rep[mask_nodes] = 0.0
        recon = self.compute(x=rep, edge_index=use_edge_index, layer_num=self.decoder_layer_emb_mask)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        loss = self.criterion_emb_mask(x_rec, x_init)
        return self.interest_recon_weight*loss

    def calc_mask_rate(self, epoch):
        total_round, increase_type, init_rate, max_rate = self.total_epoch, self.increase_type, self.min_mask_rate, self.max_mask_rate
        if increase_type == "lin":
            increase_rate = (max_rate - init_rate) / total_round
            mask_rate = init_rate + epoch * increase_rate
        elif increase_type == "exp":
            alpha = (max_rate / init_rate) ** (1 / total_round)
            mask_rate = init_rate * alpha ** epoch
        mask_rate = min(mask_rate, max_rate)    
        return mask_rate

    def get_score_matrix(self):
        all_layer = self.compute()
        U_e = all_layer[:self.users].detach().cpu() 
        V_e = all_layer[self.users:self.users+self.items].detach().cpu() 
        score_matrix = torch.matmul(U_e, V_e.t())   
        return score_matrix

    def get_kg_loss(self, batch_h, batch_t_pos, batch_t_neg, batch_r):
        h = self.user_entity_emb(batch_h)
        t_pos = self.user_entity_emb(batch_t_pos)
        t_neg = self.user_entity_emb(batch_t_neg)
        r = self.rel_emb(batch_r)
        pos_score = torch.sum(torch.pow(h + r - t_pos, 2), dim=1)     
        neg_score = torch.sum(torch.pow(h + r - t_neg, 2), dim=1)     
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)
        l2_loss = _L2_loss_mean(h) + _L2_loss_mean(r) + _L2_loss_mean(t_pos) + _L2_loss_mean(t_neg)
        loss = kg_loss + self.l2_reg_kge * l2_loss
        return loss