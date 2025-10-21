import torch
import numpy as np
import string
import copy
import multiprocessing as mp
from tqdm import tqdm
import json
import pickle as pkl
import sys

# Append modules path and import new features
sys.path.append('./modules') 
from trie_annotator import TrieAnnotator
from orwell_simplifier import OrwellSimplifier

class LSIDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_file=None, data_list=None):
        super().__init__()
        
        self.annotated = False
        self.sent_vectorized = False
        self.has_orwell_features = False # NEW FLAG
        
        # Initialize the new processors
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
                    text = [sent for sent in doc['text']] # Keep as list of strings initially
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
        with open(data_file, 'wb') as fw:
            pkl.dump(self, fw)
            
    def load_data(data_file):
        with open(data_file, 'rb') as fr:
            return pkl.load(fr)
        
    # Apply Trie, Orwell, and then standard preprocessing
    def preprocess(self, use_trie=False, use_orwell=False):
        
        for i, instance in enumerate(tqdm(self.dataset, desc="1. Trie/Orwell Preprocessing")):
            
            text_str = ' '.join(instance['text'])
            
            # 1. Extract Orwell Features
            if use_orwell:
                instance['orwell_features'] = self.orwell_simplifier.extract_features(text_str)
            else:
                instance['orwell_features'] = [0.0, 0.0, 0.0]

            # 2. Trie Annotation
            if use_trie:
                annotated_text = self.trie_annotator.annotate_text(text_str)
            else:
                annotated_text = text_str
            
            # Re-split the annotated text by space
            instance['text'] = np.array([s.strip() for s in annotated_text.split() if s.strip()])


        self.has_orwell_features = use_orwell

        for i, instance in enumerate(tqdm(self.dataset, desc="2. Standard Preprocessing")):
            text = []
            for j, sent in enumerate(instance['text']):
                if not self.sent_vectorized:
                     ppsent = sent.strip().lower().translate(str.maketrans('', '', string.punctuation))
                     if len(ppsent.split()) > 1:
                        text.append(ppsent)
                else:
                    text.append(sent) # Keep as raw string for Sent2Vec
            instance['text'] = np.array(text)


    # break each sentence string into word tokens (Only used if NOT Sent2Vec)
    def tokenize(self):
        for i, instance in enumerate(tqdm(self.dataset, desc="Tokenizing")):
            text = []
            for j, sent in enumerate(instance['text']):
                toksent = np.array(sent.strip().split())
                text.append(toksent)
            instance['text'] = np.array(text, dtype=object)
    
    # generate a vector for each sentence using Sent2Vec
    def sent_vectorize(self, sent2vec_model):      
        for i, instance in enumerate(tqdm(self.dataset, desc="Embedding sentences")):
            if sent2vec_model is not None:
                esents = sent2vec_model.embed_sentences(instance['text'].tolist())
                valid_rows = np.where(esents.sum(axis=1) != 0)[0]
                instance['text'] = esents[valid_rows]
            else:
                instance['text'] = np.zeros((0, 200))
        self.sent_vectorized = True


# unified code for generating mini batches of data for both facts and sections during train / dev / test / inference
class MiniBatch:
    def __init__(self, examples, vocab=None, label_vocab=None, schemas=None, type_map=None, node_vocab=None, edge_vocab=None, adjacency=None, hidden_size=200, max_segments=4, max_segment_size=8, num_mpath_samples=2):
        self.sent_vectorized = True if vocab is None else False
        self.annotated = True if label_vocab is not None else False
        self.sample_metapaths = True if schemas is not None else False
        
        # NEW: Check and initialize Orwell features
        self.has_orwell_features = len(examples) > 0 and 'orwell_features' in examples[0]
        if self.has_orwell_features:
            self.orwell_features = torch.zeros(len(examples), 3, dtype=torch.float) 
            
        self.max_segments = max_segments
        
        if not self.sent_vectorized:
            self.vocab = vocab
            self.max_segment_size = max_segment_size
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
        
        # Determine maximum sequence lengths
        max_len = max([len(d['text']) for d in examples])
        max_segments = min(self.max_segments, max_len)
        
        # expected shape of text tensors
        if not self.sent_vectorized:
            max_segment_size = max([len(s) for d in examples for s in d['text']]) if max_len > 0 else 1 
            max_segment_size = min(self.max_segment_size, max_segment_size)
            self.tokens = torch.zeros(len(examples), max_segments, max_segment_size, dtype=torch.long) # [D, S, W]
        else:
            self.doc_inputs = torch.zeros(len(examples), max_segments, self.sent_hidden_size) # [D, S, H]
        
        self.example_ids = []
        
        if self.annotated:
            self.labels = torch.zeros(len(examples), len(self.label_vocab)) # [D, C]
        
        for i, instance in enumerate(examples):
            if not self.sent_vectorized:
                for j, sent in enumerate(instance['text']):
                    self.tokens[i, j, :len(sent)] = torch.from_numpy(np.array([self.vocab.get(w, 0) for w in sent]))
            else:
                self.doc_inputs[i, :len(instance['text']), :] = torch.from_numpy(instance['text'])[:max_segments]
            
            self.example_ids.append(instance['id'])
            
            if self.annotated:
                label_list = torch.from_numpy(np.array([self.label_vocab[l] for l in instance['labels']]))
                self.labels[i].scatter_(0, label_list, 1.) 
            
            if self.has_orwell_features: # NEW: Fill Orwell features
                self.orwell_features[i] = torch.tensor(instance['orwell_features'], dtype=torch.float)
                
        if not self.sent_vectorized:
            self.mask = (self.tokens != 0).float() # [D, S, W]
        else:
            self.mask = (self.doc_inputs.abs().sum(dim=2) != 0).float() 
                
        if self.sample_metapaths:
            trg_node_tokens = torch.tensor([self.node_vocab[self.type_map[x]][x] for x in self.example_ids])
            self.node_tokens, self.edge_tokens = self.generate_metapaths(trg_node_tokens, self.schemas, self.adjacency, self.edge_vocab, num_samples=self.num_mpath_samples) # N * [M, D, L+1], N * [M, D, L]
    
    def generate_metapaths(self, indices, schemas, adjacency, edge_vocab, num_samples=2): 
        indices = indices.repeat(num_samples) 
        tokens, edge_tokens = [], []
        
        for i in range(len(schemas)):
            ins_tokens, ins_edge_tokens = [indices], []
            for keys in schemas[i]:
                neighbours = adjacency[keys].sample(num_neighbors=1, subset=ins_tokens[-1]).squeeze(1) 
                relations = torch.full(neighbours.shape, edge_vocab[keys[1]], dtype=torch.long) 
                
                ins_tokens.append(neighbours)
                ins_edge_tokens.append(relations)
            
            ins_tokens = torch.stack(ins_tokens, dim=1)
            ins_tokens = ins_tokens.view(num_samples, -1, ins_tokens.size(1))
            
            ins_edge_tokens = torch.stack(ins_edge_tokens, dim=1)
            ins_edge_tokens = ins_edge_tokens.view(num_samples, -1, ins_edge_tokens.size(1))
            
            tokens.append(ins_tokens)
            edge_tokens.append(ins_edge_tokens)
        
        return tokens, edge_tokens                    
    
    # automatic memory pinning for faster cpu to cuda transfer
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
    
    # transfer pinned cpu tensors to cuda
    def cuda(self, dev='cuda'):
        if not self.sent_vectorized:
            self.tokens = self.tokens.cuda(dev, non_blocking=True)
        else:
            self.doc_inputs = self.doc_inputs.cuda(dev, non_blocking=True)
        self.mask = self.mask.cuda(dev, non_blocking=True)
        if self.annotated:
            self.labels = self.labels.cuda(dev, non_blocking=True)
        if self.has_orwell_features:
            self.orwell_features = self.orwell_features.cuda(dev, non_blocking=True)
        if self.sample_metapaths:
            for i in range(len(self.node_tokens)):
                self.node_tokens[i] = self.node_tokens[i].cuda(dev, non_blocking=True)
                self.edge_tokens[i] = self.edge_tokens[i].cuda(dev, non_blocking=True)
        return self

def collate_func(examples, **kwargs):
    return MiniBatch(examples, **kwargs)