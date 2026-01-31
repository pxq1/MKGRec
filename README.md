# MKGRec: A Meta-Knowledge Guided Multi-View Recommendation Framework via Integrating LLMs

![Illustration](./pics/illustrate.png)

## Introduction

![Framework](./pics/framework.png)
To address the limitations of traditional recommender systems in dynamic interest modeling, multi-source semantic fusion, and semantic preservation during augmentation, this paper proposes a Meta-Knowledge Guided Multi-View Recommendation Framework (MKGRec) via integrating Large Language Models. The core of our approach is to construct a meta-knowledge guided multi-view contrastive learning framework. First, we utilize LLMs to parse user behavior sequences and infer latent dynamic interest evolution paths. By filtering noise through mutual information maximization, we construct a dynamic user interest graph. Next, guided by meta-knowledge, we perform cross-view semantic alignment and conflict pruning on the collaborative, interest, and knowledge views to achieve efficient fusion of multi-source heterogeneous information. Finally, we construct a semantic-preserving data augmentation pipeline via a dynamic masking strategy. Experiments on three public datasets demonstrate that our method improves Recall@50 and NDCG@50 by up to 6.81\% and 12.9\% compared to the best baselines. Ablation studies validate the key contributions of the LLM interest inference, meta-knowledge guidance, and interest augmentation learning modules. Furthermore, noise injection experiments confirm the superior robustness of our method.

## Environment Requirement

The code has been tested under Python 3.8. The required packages are as follows:

- torch == 1.13.1
- torch-geometric == 2.3.1
- scikit-learn == 1.0.2
- sentence-trainsformers == 3.0.0
- prettytable == 3.10.0
- nltk == 3.8.1

## Quick Start

- To run the code:

```
python -u train.py --dataset book-crossing
```

```
python -u train.py --dataset dbbook2014
```

```
python -u train.py --dataset ml1m
```

## Details of Important Files

- **train.py**: Main training script with dynamic masking, meta-knowledge guided contrastive learning, and multi-view fusion
- **evaluate.py**: Model evaluation with metrics (Recall, NDCG, Precision, Hit Ratio) at K=[20, 50, 100]
- **models/Model.py**: Core MKGRec model implementation with LightGCN propagation, interest reconstruction, and meta-knowledge alignment
- **models/loss_func.py**: Loss functions including BPR, InfoNCE, and SCE loss
- **utils.py**: Data loading, preprocessing, and auxiliary functions
- **User_Interest_Generation_BatchMode.py**: Generate user interest knowledge via LLMs in batch mode
- **Structration_User_Knowledge.py**: Cluster and structure user interests using TF-IDF/SentenceTransformer with mutual information filtering
- **call_llm.py**: LLM API interface for interest inference
- **config/\*.json**: Hyperparameter configurations for each dataset

## Data

Each dataset folder (e.g., `data/book-crossing/`) contains:

- **train.txt**: Training interactions (format: `user_id item_id1 item_id2 ...`)
- **test.txt**: Test interactions (same format as train.txt)
- **kg_final.txt**: Knowledge graph triplets (format: `head_id relation_id tail_id`)
- **user_interest_clustered.txt**: User-interest graph (format: `user_id interest_id`)
- **item_list.txt**: Item ID mappings with titles
- **entity_list.txt**: Entity ID mappings
- **relation_list.txt**: Relation ID mappings
- **user_list.txt**: User ID mappings

**Data Format**:

- User IDs: [0, num_users)
- Item IDs: [num_users, num_users + num_items)
- Entity IDs: [num_users + num_items, num_users + num_entities)
- Interest IDs: [num_users + num_entities - num_interests, num_users + num_entities)
