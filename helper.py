import torch
# REMOVED: import torch_sparse
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import random # ADDED: for graph sampling

from data_helper import MiniBatch
from modules.kdtree_retrieval import KDTreeRetriever

def generate_vocabs(train_data, label_data, limit=30000, thresh=1):
    vocab = None
    
    # Only build vocab if we're NOT using sent2vec (i.e., text is not vectorized)
    if not train_data.sent_vectorized:
        print("Building word vocabulary...")
        freqs = defaultdict(int)
        
        for instance in tqdm(train_data.dataset + label_data.dataset, desc="Counting word frequencies"):
            # instance['text'] should be a list of sentences (strings)
            if isinstance(instance['text'], (list, np.ndarray)):
                for sent in instance['text']:
                    if isinstance(sent, str):
                        for word in sent.split():
                            freqs[word] += 1
                    elif isinstance(sent, np.ndarray):
                        # Already tokenized
                        for word in sent:
                            freqs[str(word)] += 1
        
        # Build vocabulary with special tokens
        vocab_words = [w for w, f in freqs.items() if f >= thresh]
        vocab_words = sorted(vocab_words)[:limit] if limit else vocab_words
        
        vocab = {'<PAD>': 0, '<UNK>': 1}  # Reserve 0 and 1
        vocab.update({word: idx + 2 for idx, word in enumerate(vocab_words)})
        
        print(f"✓ Vocabulary size: {len(vocab)} words")
    else:
        print("✓ Using sent2vec embeddings (no vocabulary needed)")
    
    # Build label vocabulary
    label_vocab = {}
    for instance in tqdm(label_data.dataset, desc="Creating label vocabulary"):
        label_vocab[instance['id']] = len(label_vocab)
    
    print(f"✓ Label vocabulary size: {len(label_vocab)} labels")
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
        
        # CRITICAL DEBUG: Print first batch with more precision
        if i == 0:
            print(f"\n{'='*60}")
            print(f"FIRST BATCH DIAGNOSTICS ({'TRAIN' if train else 'EVAL'})")
            print(f"{'='*60}")
            print(f"Loss value: {loss.item() if loss is not None else 'None'}")
            print(f"Loss requires_grad: {loss.requires_grad if loss is not None else 'N/A'}")
            print(f"\nPredictions (probabilities after threshold):")
            print(f"  Shape: {predictions.shape}")
            print(f"  Sum: {predictions.sum().item()}")
            print(f"  Mean: {predictions.mean().item():.6f}")
            print(f"  Max: {predictions.max().item():.6f}")
            print(f"  Min: {predictions.min().item():.6f}")
            print(f"  Num > 0.20: {(predictions > 0.20).sum().item()}")
            print(f"  Num > 0.10: {(predictions > 0.10).sum().item()}")
            print(f"  Num > 0.05: {(predictions > 0.05).sum().item()}")
            print(f"\nLabels (ground truth):")
            print(f"  Shape: {fact_batch.labels.shape}")
            print(f"  Sum (total positive labels): {fact_batch.labels.sum().item()}")
            print(f"  Avg labels per doc: {fact_batch.labels.sum().item() / fact_batch.labels.size(0):.2f}")
            print(f"\nModel outputs check:")
            print(f"  Fact embeddings mean: {fact_attr_hidden.mean().item():.6f}")
            print(f"  Fact embeddings std: {fact_attr_hidden.std().item():.6f}")
            print(f"  Sec embeddings mean: {sec_struct_hidden.mean().item():.6f}")
            print(f"  Sec embeddings std: {sec_struct_hidden.std().item():.6f}")
            print(f"{'='*60}\n")
        
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        if not infer:
            batch_loss = loss.item() if loss is not None else 0.0
            labels_on_device = fact_batch.labels.to(dev)
            metrics(predictions, labels_on_device, loss=batch_loss)
        else:
            # ... infer code
            pass

    return metrics.calculate_metrics() if not infer else outputs

class MultiLabelMetrics(torch.nn.Module):
    def __init__(self, num_classes, dev='cuda', loss=True):
        super().__init__()
        
        # Use float tensors to avoid integer division issues
        self.register_buffer('match', torch.zeros(num_classes, dtype=torch.float))
        self.register_buffer('predictions', torch.zeros(num_classes, dtype=torch.float))
        self.register_buffer('labels', torch.zeros(num_classes, dtype=torch.float))
        
        self.run_jacc = 0.0 
        self.counter = 0 
        
        if loss:
            self.run_loss = 0.0
            self.has_loss = True
        else:
            self.has_loss = False
        
        self.dev = dev
        self.to(dev) 
    
    def forward(self, predictions, labels, loss=None):
        # Ensure inputs are float and on same device
        predictions = predictions.float()
        labels = labels.float()
        
        # Ensure metric tensors are on correct device
        device = predictions.device
        if self.match.device != device:
            self.match = self.match.to(device)
            self.predictions = self.predictions.to(device)
            self.labels = self.labels.to(device)
        
        # Calculate matches (element-wise multiplication, then sum across batch)
        match = predictions * labels  # [B, C]
        
        self.match += match.sum(dim=0)  # Sum across batch dimension
        self.predictions += predictions.sum(dim=0)
        self.labels += labels.sum(dim=0)
        
        # Jaccard per document
        intersection = torch.logical_and(predictions.bool(), labels.bool()).sum(dim=1).float()
        union = torch.logical_or(predictions.bool(), labels.bool()).sum(dim=1).float()
        
        # Avoid division by zero for Jaccard
        jaccard_scores = torch.where(union > 0, intersection / union, torch.zeros_like(intersection))
        
        self.run_jacc += jaccard_scores.sum().item()
        self.counter += predictions.size(0) 

        if self.has_loss and loss is not None:
            self.run_loss += loss
        
    def refresh(self):
        self.match.fill_(0)
        self.predictions.fill_(0)
        self.labels.fill_(0)
        self.run_jacc = 0.0
        self.counter = 0
        if self.has_loss:
            self.run_loss = 0.0
        return self
            
    def calculate_metrics(self, refresh=True):
        # Move to CPU for calculation to avoid any device issues
        match = self.match.cpu()
        preds = self.predictions.cpu()
        lbls = self.labels.cpu()
        
        # Micro metrics
        match_total = match.sum().item()
        preds_total = preds.sum().item()
        labels_total = lbls.sum().item()
        
        # Avoid division by zero
        self.micro_prec = match_total / preds_total if preds_total > 0 else 0.0
        self.micro_rec = match_total / labels_total if labels_total > 0 else 0.0
        self.micro_f1 = 0.0 if (self.micro_prec + self.micro_rec) == 0 else 2 * self.micro_prec * self.micro_rec / (self.micro_prec + self.micro_rec)
        
        # Macro metrics - per class
        prec = torch.where(preds > 0, match / preds, torch.zeros_like(match))
        rec = torch.where(lbls > 0, match / lbls, torch.zeros_like(match))
        
        # F1 per class
        f1 = torch.where(
            (prec + rec) > 0,
            2 * prec * rec / (prec + rec),
            torch.zeros_like(prec)
        )
        
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
