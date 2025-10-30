import torch
import numpy as np
import string
import copy
from tqdm import tqdm
import json
import pickle as pkl
import sys
import os 
import random 

# Ensure modules directory is on path
sys.path.append('./modules') 
from trie_annotator import TrieAnnotator
from orwell_simplifier import OrwellSimplifier

class LSIDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_file=None, data_list=None):
        super().__init__()
        
        self.annotated = False
        self.sent_vectorized = False
        self.has_orwell_features = False
        
        self.trie_annotator = TrieAnnotator()
        self.orwell_simplifier = OrwellSimplifier()
        
        if data_list is not None:
            self.dataset = copy.deepcopy(data_list)
            for instance in tqdm(self.dataset, desc="Loading data from list"):
                instance['text'] = instance['text']
                if 'labels' in instance:
                    self.annotated = True
                    instance['labels'] = np.array(instance['labels'])
        
        elif jsonl_file is not None:
            self.dataset = []
            with open(jsonl_file) as fr:
                for line in tqdm(fr, desc="Loading data from file"):
                    doc = json.loads(line)
                    text = [sent for sent in doc['text']]
                    newdoc = {'id': doc['id'], 'text': text}
                    if 'labels' in doc:
                        self.annotated = True
                        labels = np.array(doc['labels'])
                        newdoc['labels'] = labels
                    self.dataset.append(newdoc)
                    
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        return self.dataset[index]
    
    def save_data(self, data_file):
        os.makedirs(os.path.dirname(data_file), exist_ok=True) 
        
        with open(data_file, 'wb') as fw:
            pkl.dump(self, fw)
            
    def load_data(data_file):
        with open(data_file, 'rb') as fr:
            return pkl.load(fr)
        
    def preprocess(self, use_trie=False, use_orwell=False):
        """Fixed preprocessing that preserves sentence structure"""
        
        for i, instance in enumerate(tqdm(self.dataset, desc="1. Trie/Orwell Preprocessing")):
            # instance['text'] is already a list of sentences from JSONL loading
            sentences = instance['text']
            
            # Ensure it's a list of strings
            if isinstance(sentences, np.ndarray):
                sentences = sentences.tolist()
            if not isinstance(sentences, list):
                sentences = [str(sentences)]
            
            # Join for Orwell feature extraction (needs full document)
            full_text = ' '.join(str(s) for s in sentences)
            
            # Extract Orwell features from the full text
            if use_orwell:
                instance['orwell_features'] = self.orwell_simplifier.extract_features(full_text)
            else:
                instance['orwell_features'] = [0.0, 0.0, 0.0]
            
            # Apply Trie annotation to EACH SENTENCE individually
            if use_trie:
                annotated_sentences = []
                for sent in sentences:
                    annotated_sent = self.trie_annotator.annotate_text(str(sent))
                    if annotated_sent.strip():  # Only add non-empty sentences
                        annotated_sentences.append(annotated_sent)
            else:
                annotated_sentences = [str(s) for s in sentences if str(s).strip()]
            
            # Store as list of sentences (NOT words!)
            instance['text'] = annotated_sentences
        
        self.has_orwell_features = use_orwell
        
        # Standard preprocessing (lowercase, remove punctuation)
        for i, instance in enumerate(tqdm(self.dataset, desc="2. Standard Preprocessing")):
            processed_sentences = []
            
            for sent in instance['text']:
                if not self.sent_vectorized:
                    # Clean the sentence: lowercase and remove punctuation
                    cleaned = sent.strip().lower()
                    # Remove punctuation but keep spaces
                    cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))
                    # Only keep sentences with at least 1 word
                    if len(cleaned.split()) >= 1:
                        processed_sentences.append(cleaned)
                else:
                    # Already vectorized, just pass through
                    processed_sentences.append(sent)
            
            # Ensure we have at least one sentence (fallback)
            if len(processed_sentences) == 0:
                processed_sentences = ['unknown document']
            
            instance['text'] = processed_sentences  # Keep as Python list for now


    def tokenize(self):
        """Fixed tokenization"""
        for i, instance in enumerate(tqdm(self.dataset, desc="Tokenizing")):
            text = []
            for j, sent in enumerate(instance['text']):
                toksent = np.array(sent.strip().split())
                text.append(toksent)
            instance['text'] = np.array(text, dtype=object)  # Keep as array of arrays
    
    def sent_vectorize(self, sent2vec_model):
        """Convert sentences to vectors using sent2vec"""
        for i, instance in enumerate(tqdm(self.dataset, desc="Embedding sentences")):
            if sent2vec_model is not None:
                # Ensure text is a list of strings
                sentences = instance['text']
                if isinstance(sentences, np.ndarray):
                    sentences = sentences.tolist()
                
                # Convert to list of strings
                sentences = [str(s) for s in sentences]
                
                # Embed sentences
                try:
                    esents = sent2vec_model.embed_sentences(sentences)
                    
                    # Check if we got valid embeddings
                    if esents.shape[0] == 0 or esents.shape[1] == 0:
                        print(f"WARNING: Zero embeddings for document {instance.get('id', i)}")
                        instance['text'] = np.zeros((1, 200))  # At least keep 1 dummy vector
                    else:
                        # Only filter out completely zero vectors (rare)
                        # But be more lenient - keep vectors with small values too
                        valid_rows = np.where(np.abs(esents).sum(axis=1) > 1e-6)[0]
                        
                        if len(valid_rows) == 0:
                            # If all vectors are zero, keep the first one anyway
                            print(f"WARNING: All zero vectors for document {instance.get('id', i)}, keeping first vector")
                            instance['text'] = esents[:1]
                        else:
                            instance['text'] = esents[valid_rows]
                except Exception as e:
                    print(f"ERROR embedding document {instance.get('id', i)}: {e}")
                    instance['text'] = np.zeros((1, 200))
            else:
                # No sent2vec model - create dummy embeddings
                print("WARNING: No sent2vec model, creating zero embeddings")
                instance['text'] = np.zeros((len(instance['text']), 200))
        
        self.sent_vectorized = True


class MiniBatch:
    def __init__(self, examples, vocab=None, label_vocab=None, schemas=None, type_map=None, node_vocab=None, edge_vocab=None, adjacency=None, hidden_size=200, max_segments=4, max_segment_size=8, num_mpath_samples=2):
        self.sent_vectorized = True if vocab is None else False
        self.annotated = True if label_vocab is not None else False
        self.sample_metapaths = True if schemas is not None else False
        
        self.has_orwell_features = len(examples) > 0 and 'orwell_features' in examples[0]
        if self.has_orwell_features:
            self.orwell_features = torch.zeros(len(examples), 3, dtype=torch.float) 
            
        self.max_segments = max_segments
        
        if not self.sent_vectorized:
            self.vocab = vocab
            # Handle None max_segment_size
            self.max_segment_size = max_segment_size if max_segment_size is not None else 128
        else:
            self.sent_hidden_size = hidden_size
            
        if self.annotated:
            self.label_vocab = label_vocab
            
        if self.sample_metapaths:
            self.schemas = schemas
            self.type_map = type_map
            self.node_vocab = node_vocab
            self.edge_vocab = edge_vocab
            self.adjacency = adjacency
            self.num_mpath_samples = num_mpath_samples
        
        # Calculate max length from examples
        max_len = max([len(d['text']) for d in examples]) if len(examples) > 0 else 1
        max_segments = min(self.max_segments, max_len)
        
        self.example_ids = []
        
        if self.annotated:
            self.labels = torch.zeros(len(examples), len(self.label_vocab))
        
        # Initialize tensors based on vectorization mode
        if not self.sent_vectorized:
            # Calculate actual max segment size from data
            actual_max_segment_size = 1
            if len(examples) > 0 and max_len > 0:
                for d in examples:
                    for s in d['text']:
                        if isinstance(s, str):
                            actual_max_segment_size = max(actual_max_segment_size, len(s.split()))
                        elif isinstance(s, np.ndarray):
                            actual_max_segment_size = max(actual_max_segment_size, len(s))
            
            # Use the minimum of configured and actual
            max_segment_size = min(self.max_segment_size, actual_max_segment_size)
            self.tokens = torch.zeros(len(examples), max_segments, max_segment_size, dtype=torch.long)
        else:
            self.doc_inputs = torch.zeros(len(examples), max_segments, self.sent_hidden_size)
        
        # Process examples
        for i, instance in enumerate(examples):
            if not self.sent_vectorized:
                for j, sent in enumerate(instance['text'][:max_segments]):
                    words = []
                    
                    if isinstance(sent, str):
                        words = sent.split()
                    elif isinstance(sent, np.ndarray):
                        words = [str(w) for w in sent]
                    
                    # Truncate to max_segment_size
                    words = words[:max_segment_size]
                    
                    # Get word IDs, defaulting to <UNK> (1) if not in vocab
                    token_ids = [self.vocab.get(word, 1) for word in words]
                    
                    if len(token_ids) > 0:
                        self.tokens[i, j, :len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
            else:
                # Using sent2vec embeddings
                self.doc_inputs[i, :len(instance['text']), :] = torch.from_numpy(instance['text'])[:max_segments]
            
            self.example_ids.append(instance['id'])
            
            if self.annotated:
                label_indices = [self.label_vocab[l] for l in instance['labels']]
                label_list = torch.as_tensor(label_indices, dtype=torch.long)
                self.labels[i].scatter_(0, label_list, 1.) 
            
            if self.has_orwell_features:
                self.orwell_features[i] = torch.tensor(instance['orwell_features'], dtype=torch.float)
        
        # Create mask AFTER processing all examples
        if not self.sent_vectorized:
            self.mask = (self.tokens != 0).float()
        else:
            self.mask = (self.doc_inputs.abs().sum(dim=2) != 0).float()
        
        # Generate metapaths if needed
        if self.sample_metapaths:
            trg_node_tokens = torch.tensor([self.node_vocab[self.type_map[x]][x] for x in self.example_ids])
            self.node_tokens, self.edge_tokens = self.generate_metapaths(trg_node_tokens, self.schemas, self.adjacency, self.edge_vocab, num_samples=self.num_mpath_samples)
            
    # Manual implementation of neighbor sampling (replaces torch_sparse.sample())
    def generate_metapaths(self, indices, schemas, adjacency, edge_vocab, num_samples=2): 
        
        indices = indices.repeat(num_samples) # [M*D,]
        tokens, edge_tokens = [], []
        
        for i in range(len(schemas)):
            ins_tokens, ins_edge_tokens = [indices], []
            
            for keys in schemas[i]:
                current_indices = ins_tokens[-1].tolist()
                neighbours_list = []
                
                # Perform manual random neighbor sampling for each node index
                for index in current_indices:
                    node_type = keys[0]
                    
                    # 1. Reverse Lookup (Find node name from token ID)
                    node_name = next((name for name, idx in self.node_vocab[node_type].items() if idx == index), None)
                    
                    if node_name is None:
                        # If node ID is corrupted, sample identity
                        sampled_index = index
                    else:
                        # 2. Get Neighbors from Adjacency List
                        possible_neighbors = adjacency.get(keys, {}).get(node_name, [])
                        
                        if possible_neighbors:
                            # 3. Sample Neighbor
                            sampled_name = random.choice(possible_neighbors)
                            # 4. Forward Lookup (Get new token ID)
                            sampled_index = self.node_vocab[keys[2]][sampled_name]
                        else:
                            # If no neighbors, sample the node itself (identity)
                            sampled_index = index 

                    neighbours_list.append(sampled_index)
                
                neighbours = torch.tensor(neighbours_list, dtype=torch.long) # [M*D,]
                
                # --- FIX: CLAMP INDICES TO PREVENT CUDA CRASH (device-side assert) ---
                target_node_type = keys[2]
                max_valid_index = len(self.node_vocab[target_node_type]) - 1
                
                # Clamp all indices to ensure they are within the [0, max_valid_index] range
                neighbours = torch.clamp(neighbours, min=0, max=max_valid_index) 
                # --------------------------------------------------------------------
                
                relations = torch.full(neighbours.shape, edge_vocab[keys[1]], dtype=torch.long) # [M*D,]
                
                ins_tokens.append(neighbours)
                ins_edge_tokens.append(relations)
            
            ins_tokens = torch.stack(ins_tokens, dim=1)
            ins_tokens = ins_tokens.view(num_samples, -1, ins_tokens.size(1))
            
            ins_edge_tokens = torch.stack(ins_edge_tokens, dim=1)
            ins_edge_tokens = ins_edge_tokens.view(num_samples, -1, ins_edge_tokens.size(1))
            
            tokens.append(ins_tokens)
            edge_tokens.append(ins_edge_tokens)
        
        return tokens, edge_tokens                    
    
    def pin_memory(self):
        if not self.sent_vectorized:
            self.tokens.pin_memory()
        else:
            self.doc_inputs.pin_memory()
        self.mask.pin_memory()
        if self.annotated:
            self.labels.pin_memory()
        if self.has_orwell_features:
            self.orwell_features.pin_memory()
        if self.sample_metapaths:
            for i in range(len(self.node_tokens)):
                self.node_tokens[i].pin_memory()
                self.edge_tokens[i].pin_memory()
        return self
    
    def to_device(self, dev):
        if not self.sent_vectorized:
            self.tokens = self.tokens.to(dev, non_blocking=True)
        else:
            self.doc_inputs = self.doc_inputs.to(dev, non_blocking=True)
        
        self.mask = self.mask.to(dev, non_blocking=True)
        
        if self.annotated:
            self.labels = self.labels.to(dev, non_blocking=True)
        if self.has_orwell_features:
            self.orwell_features = self.orwell_features.to(dev, non_blocking=True)
            
        if self.sample_metapaths:
            for i in range(len(self.node_tokens)):
                self.node_tokens[i] = self.node_tokens[i].to(dev, non_blocking=True)
                self.edge_tokens[i] = self.edge_tokens[i].to(dev, non_blocking=True)
        return self

def collate_func(examples, **kwargs):
    return MiniBatch(examples, **kwargs)
