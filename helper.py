import torch
# REMOVED: import torch_sparse
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import random # ADDED: for graph sampling

from data_helper import MiniBatch
from modules.kdtree_retrieval import KDTreeRetriever

def generate_vocabs(train_data, label_data, limit=30000, thresh=1):
    if not train_data.sent_vectorized:
        freqs = defaultdict(int)
        for instance in tqdm(train_data.dataset + label_data.dataset, desc="Creating vocabulary"):
            if isinstance(instance['text'], np.ndarray) and instance['text'].ndim == 1:
                 for word in instance['text']:
                     freqs[word] += 1
            else: 
                for sent in instance['text']:
                    for word in sent.split():
                        freqs[word] += 1
                        
        vocab_set = set(w for w, f in freqs.items() if f >= thresh)
        vocab = {k: i for i, k in enumerate(vocab_set)}
    else:
        vocab = None
    
    label_vocab = {}
    for instance in tqdm(label_data.dataset, desc="Creating label vocabulary"):
        label_vocab[instance['id']] = len(label_vocab)

    return vocab, label_vocab

def generate_graph(label_vocab, type_map, label_tree_edges, cit_net_edges, label_name='section'):
    node_vocab = defaultdict(dict) 
    node_vocab[label_name] = label_vocab 

    edge_vocab = {}
    edge_indices = defaultdict(list) 
    
    # Adjacency list: adjacency[(start_type, relation, end_type)] = {'start_node_name': ['end_node_name', ...]}
    adjacency = defaultdict(lambda: defaultdict(list)) 

    for (node_a, edge_type, node_b) in label_tree_edges + cit_net_edges:
        node_a_type, node_b_type = type_map[node_a], type_map[node_b]

        if edge_type not in edge_vocab:
            edge_vocab[edge_type] = len(edge_vocab)

        if node_a not in node_vocab[node_a_type]:
            node_vocab[node_a_type][node_a] = len(node_vocab[node_a_type])
        if node_b not in node_vocab[node_b_type]:
            node_vocab[node_b_type][node_b] = len(node_vocab[node_b_type])
            
        # Store connection using node names (strings)
        adjacency[(node_a_type, edge_type, node_b_type)][node_a].append(node_b)

    num_nodes = {ntype: len(nodes) for ntype, nodes in node_vocab.items()}
    
    # edge_indices is returned empty as it's no longer used, but function signature is kept
    return node_vocab, edge_vocab, edge_indices, adjacency

def generate_label_weights(train_data, label_vocab, dev='cuda:0', scheme="tws", thresh=10.):
    pos = torch.zeros(len(label_vocab)) 
    
    for instance in tqdm(train_data, desc="Generating label weights"):
        if 'labels' in instance and instance['labels'] is not None:
             for l in instance['labels']:
                if l in label_vocab:
                    pos[label_vocab[l]] += 1
                    
    pos_safe = pos.clone()
    pos_safe[pos_safe == 0] = 1 
    
    if scheme == 'tws':
        weights = torch.clamp(pos.max() / pos_safe, max=thresh).to(dev)
    else: 
        weights = (len(train_data) / pos_safe).to(dev)
        
    return weights

def train_dev_pass(model, optimizer, fact_loader, sec_batch, metrics=None, pred_threshold=None, train=False, infer=False, label_vocab=False, kdtree: KDTreeRetriever = None): 
    model.train() if train else model.eval()
    
    dev = next(model.parameters()).device 
    
    if infer:
        outputs = []
        inv_label_vocab = {v: k for k, v in label_vocab.items()}

    for i, fact_batch in enumerate(tqdm(fact_loader, desc="Flowing data through model")):    
        torch.cuda.empty_cache()

        fact_batch = fact_batch.to_device(dev)
        sec_batch = sec_batch.to_device(dev)
        
        loss, predictions, sec_struct_hidden, fact_attr_hidden = model(fact_batch, sec_batch, pthresh=pred_threshold)
        
        if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        if not infer:
            batch_loss = loss.item() if loss is not None else 0.0
            
            labels_on_device = fact_batch.labels.to(dev)
            
            metrics(predictions, labels_on_device, loss=batch_loss)
        
        else:
            final_predictions = predictions
            if kdtree is not None:
                fact_embeddings_np = fact_attr_hidden.cpu().numpy()
                candidate_indices_batch = kdtree.batch_query(fact_embeddings_np, k=kdtree.k_neighbors) 

                final_predictions = torch.zeros_like(predictions, dtype=torch.float)
                for doc_idx in range(predictions.size(0)):
                    candidates = candidate_indices_batch[doc_idx]
                    
                    candidate_mask = torch.zeros(predictions.size(1), device=predictions.device)
                    candidate_mask[candidates] = 1.0
                    
                    final_predictions[doc_idx] = predictions[doc_idx] * candidate_mask

            for doc_idx, instance_preds in enumerate(final_predictions):
                pred_list_indices = torch.nonzero(instance_preds, as_tuple=False).squeeze(1).cpu().tolist()
                pred_list = [inv_label_vocab[idx] for idx in pred_list_indices]
                outputs.append({'id': fact_batch.example_ids[doc_idx], 'predictions': pred_list})


    return metrics.calculate_metrics() if not infer else outputs

class MultiLabelMetrics(torch.nn.Module):
    def __init__(self, num_classes, dev='cuda', loss=True):
        super().__init__()
        
        self.match = torch.zeros(num_classes) 
        self.predictions = torch.zeros(num_classes) 
        self.labels = torch.zeros(num_classes) 
        
        self.run_jacc = 0 
        self.counter = 0 
        
        if loss:
            self.run_loss = 0.0
            self.has_loss = True
        else:
            self.has_loss = False
        
        self.dev = dev
        self.to(dev) 
    
    def forward(self, predictions, labels, loss=None):
        # Ensure internal metric tensors are on the correct device (GPU/CPU)
        device = predictions.device
        if self.match.device != device:
            self.match = self.match.to(device)
            self.predictions = self.predictions.to(device)
            self.labels = self.labels.to(device)
            
        match = predictions * labels 
        
        self.match += match.sum(dim=0)
        self.predictions += predictions.sum(dim=0)
        self.labels += labels.sum(dim=0)
        
        intersection = torch.logical_and(predictions, labels).sum(dim=1).float()
        union = torch.logical_or(predictions, labels).sum(dim=1).float()
        jaccard_scores = intersection / (union + 1e-10)
        
        self.run_jacc += jaccard_scores.sum().item()
        self.counter += predictions.size(0) 

        if self.has_loss and loss is not None:
            self.run_loss += loss
        
    def refresh(self):
        self.match.fill_(0)
        self.predictions.fill_(0)
        self.labels.fill_(0)
        self.run_jacc = 0
        self.counter = 0
        if self.has_loss:
            self.run_loss = 0.0
        return self
            
    def calculate_metrics(self, refresh=True):
        # Micro metrics calculation
        match_total = self.match.sum().item()
        preds_total = self.predictions.sum().item()
        labels_total = self.labels.sum().item()
        
        # FIX: Check for zero division before calculating micro-P and micro-R
        self.micro_prec = match_total / preds_total if preds_total > 0 else 0.0
        self.micro_rec = match_total / labels_total if labels_total > 0 else 0.0 # Solves ZeroDivisionError
        self.micro_f1 = 0.0 if self.micro_prec + self.micro_rec == 0 else 2 * self.micro_prec * self.micro_rec / (self.micro_prec + self.micro_rec)
        
        # Macro metrics calculation
        prec = self.match / self.predictions
        rec = self.match / self.labels
        
        # Handle NaN from macro division by zero
        prec[prec.isnan()] = 0.0
        rec[rec.isnan()] = 0.0
        
        f1 = 2 * prec * rec / (prec + rec)
        f1[f1.isnan()] = 0.0
        
        self.macro_prec = prec.mean().item()
        self.macro_rec = rec.mean().item()
        self.macro_f1 = f1.mean().item()
        
        self.jacc = self.run_jacc / self.counter if self.counter > 0 else 0.0
        
        if self.has_loss:
            self.loss = self.run_loss / self.counter if self.counter > 0 else 0.0
        else:
            self.loss = 0.0 
        
        if refresh:
            self.refresh()
        
        return self
