"""
Enhanced LeSICiN Training Script for VSCode
Simplified standalone version with Trie and Orwell features
"""
import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import jsonlines
import sys
from pathlib import Path

# Add modules to path
sys.path.append('modules')

from modules.trie_annotator import TrieAnnotator
from modules.orwell_simplifier import OrwellSimplifier
from modules.kdtree_retrieval import KDTreeRetriever


class LegalDataset(Dataset):
    """Dataset with Trie and Orwell preprocessing"""
    
    def __init__(self, data_path, word_to_idx=None, label_to_idx=None, 
                 max_segments=50, max_words=100, use_trie=True, use_orwell=True):
        self.data = []
        with jsonlines.open(data_path) as reader:
            for obj in reader:
                self.data.append(obj)
        
        self.max_segments = max_segments
        self.max_words = max_words
        self.use_trie = use_trie
        self.use_orwell = use_orwell
        
        # Initialize preprocessors
        if use_trie:
            self.trie = TrieAnnotator()
        if use_orwell:
            self.orwell = OrwellSimplifier()
        
        # Build or use provided vocabularies
        self.word_to_idx = word_to_idx
        self.label_to_idx = label_to_idx
        
        if word_to_idx is None or label_to_idx is None:
            self._build_vocab()
    
    def _build_vocab(self):
        """Build word and label vocabularies"""
        print("Building vocabularies...")
        word_freq = {}
        labels_set = set()
        
        for item in tqdm(self.data, desc="Processing"):
            text = item['text']
            if self.use_trie:
                text = self.trie.annotate_text(text)
            
            for sent in text:
                for word in sent.lower().split():
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            if 'labels' in item and item['labels']:
                labels_set.update(item['labels'])
        
        # Build word vocabulary
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for word, freq in sorted(word_freq.items(), key=lambda x: -x[1]):
            if freq >= 3 and len(self.word_to_idx) < 10000:
                self.word_to_idx[word] = len(self.word_to_idx)
        
        # Build label vocabulary
        self.label_to_idx = {}
        for label in sorted(labels_set):
            self.label_to_idx[label] = len(self.label_to_idx)
        
        print(f"✓ Vocabulary: {len(self.word_to_idx)} words, {len(self.label_to_idx)} labels")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process text
        text = item['text']
        if self.use_trie:
            text = self.trie.annotate_text(text)
        
        # Encode text
        encoded = []
        for sent in text[:self.max_segments]:
            words = sent.lower().split()[:self.max_words]
            word_ids = [self.word_to_idx.get(w, 1) for w in words]
            # Pad sentence
            word_ids += [0] * (self.max_words - len(word_ids))
            encoded.append(word_ids)
        
        # Pad document
        while len(encoded) < self.max_segments:
            encoded.append([0] * self.max_words)
        
        # Get Orwell features
        if self.use_orwell:
            orwell_feats = self.orwell.extract_features(item['text'])
        else:
            orwell_feats = [0.0, 0.0, 0.0]
        
        # Encode labels
        labels = torch.zeros(len(self.label_to_idx))
        if 'labels' in item and item['labels']:
            for label in item['labels']:
                if label in self.label_to_idx:
                    labels[self.label_to_idx[label]] = 1.0
        
        return {
            'text': torch.LongTensor(encoded),
            'orwell_features': torch.FloatTensor(orwell_feats),
            'labels': labels,
            'id': item.get('id', str(idx))
        }


class SimpleHAN(nn.Module):
    """Hierarchical Attention Network with Orwell features"""
    
    def __init__(self, vocab_size, hidden_size, num_labels, orwell_dim=3):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Word-level
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.word_lstm = nn.LSTM(hidden_size, hidden_size//2, bidirectional=True, batch_first=True)
        self.word_attention = nn.Linear(hidden_size, 1)
        
        # Sentence-level
        self.sent_lstm = nn.LSTM(hidden_size, hidden_size//2, bidirectional=True, batch_first=True)
        self.sent_attention = nn.Linear(hidden_size, 1)
        
        # Orwell feature projection
        self.orwell_proj = nn.Linear(orwell_dim, hidden_size//4)
        
        # Classifier
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_size + hidden_size//4, num_labels)
    
    def forward(self, text, orwell_features):
        batch_size, num_sents, num_words = text.size()
        
        # Reshape for word-level processing
        text = text.view(batch_size * num_sents, num_words)
        emb = self.embedding(text)
        
        # Word-level LSTM
        word_out, _ = self.word_lstm(emb)
        
        # Word-level attention
        word_attn = torch.softmax(self.word_attention(word_out), dim=1)
        sent_repr = torch.sum(word_out * word_attn, dim=1)
        
        # Reshape back to document level
        sent_repr = sent_repr.view(batch_size, num_sents, -1)
        
        # Sentence-level LSTM
        sent_out, _ = self.sent_lstm(sent_repr)
        
        # Sentence-level attention
        sent_attn = torch.softmax(self.sent_attention(sent_out), dim=1)
        doc_repr = torch.sum(sent_out * sent_attn, dim=1)
        
        # Project Orwell features
        orwell_proj = torch.relu(self.orwell_proj(orwell_features))
        
        # Concatenate
        combined = torch.cat([doc_repr, orwell_proj], dim=-1)
        combined = self.dropout(combined)
        
        # Classify
        logits = self.classifier(combined)
        return logits


def compute_metrics(preds, labels):
    """Compute macro and micro metrics"""
    preds = preds.numpy()
    labels = labels.numpy()
    
    # Macro metrics
    macro_p, macro_r, macro_f1 = [], [], []
    
    for i in range(labels.shape[1]):
        tp = np.sum((preds[:, i] == 1) & (labels[:, i] == 1))
        fp = np.sum((preds[:, i] == 1) & (labels[:, i] == 0))
        fn = np.sum((preds[:, i] == 0) & (labels[:, i] == 1))
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        macro_p.append(p)
        macro_r.append(r)
        macro_f1.append(f1)
    
    # Micro metrics
    tp_total = np.sum((preds == 1) & (labels == 1))
    fp_total = np.sum((preds == 1) & (labels == 0))
    fn_total = np.sum((preds == 0) & (labels == 1))
    
    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
    
    # Jaccard
    intersection = np.sum((preds == 1) & (labels == 1), axis=1)
    union = np.sum((preds == 1) | (labels == 1), axis=1)
    jaccard = np.mean(intersection / (union + 1e-10))
    
    return {
        'macro': {
            'precision': np.mean(macro_p) * 100,
            'recall': np.mean(macro_r) * 100,
            'f1': np.mean(macro_f1) * 100
        },
        'micro': {
            'precision': micro_p * 100,
            'recall': micro_r * 100,
            'f1': micro_f1 * 100
        },
        'jaccard': jaccard * 100
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        text = batch['text'].to(device)
        orwell_feats = batch['orwell_features'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits = model(text, orwell_feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, threshold=0.5):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            text = batch['text'].to(device)
            orwell_feats = batch['orwell_features'].to(device)
            labels = batch['labels']
            
            logits = model(text, orwell_feats)
            preds = (torch.sigmoid(logits) >= threshold).float().cpu()
            
            all_preds.append(preds)
            all_labels.append(labels)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    metrics = compute_metrics(all_preds, all_labels)
    return metrics


def main():
    """Main training function"""
    print("="*80)
    print("Enhanced LeSICiN Training")
    print("="*80)
    
    # Load config
    with open('configs/hyperparams.json', 'r') as f:
        config = json.load(f)
    
    with open('configs/data_paths.json', 'r') as f:
        data_paths = json.load(f)
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\nConfiguration:")
    print(f"  Trie annotation: {config['use_trie']}")
    print(f"  Orwell features: {config['use_orwell']}")
    print(f"  Batch size: {config['train_bs']}")
    print(f"  Epochs: {config['num_epoch']}")
    
    # Load datasets
    print("\n" + "="*80)
    print("Loading Data")
    print("="*80)
    
    print("\nLoading training data...")
    train_dataset = LegalDataset(
        data_paths['train_src'],
        max_segments=config['max_segments'],
        max_words=config['max_segment_size'],
        use_trie=config['use_trie'],
        use_orwell=config['use_orwell']
    )
    
    print("\nLoading validation data...")
    dev_dataset = LegalDataset(
        data_paths['dev_src'],
        word_to_idx=train_dataset.word_to_idx,
        label_to_idx=train_dataset.label_to_idx,
        max_segments=config['max_segments'],
        max_words=config['max_segment_size'],
        use_trie=config['use_trie'],
        use_orwell=config['use_orwell']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_bs'],
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config['dev_bs'],
        shuffle=False,
        num_workers=0
    )
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset):,} instances")
    print(f"  Validation: {len(dev_dataset):,} instances")
    print(f"  Vocabulary: {len(train_dataset.word_to_idx):,} words")
    print(f"  Labels: {len(train_dataset.label_to_idx)} sections")
    
    # Build model
    print("\n" + "="*80)
    print("Building Model")
    print("="*80)
    
    model = SimpleHAN(
        vocab_size=len(train_dataset.word_to_idx),
        hidden_size=config['hidden_size'],
        num_labels=len(train_dataset.label_to_idx),
        orwell_dim=3
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['opt_lr'],
        weight_decay=config['opt_wt_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['sch_factor'],
        patience=config['sch_patience'],
        verbose=True
    )
    
    # Training loop
    print("\n" + "="*80)
    print("Training")
    print("="*80)
    
    best_f1 = 0
    best_epoch = 0
    
    for epoch in range(config['num_epoch']):
        print(f"\nEpoch {epoch + 1}/{config['num_epoch']}")
        print("-" * 80)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        dev_metrics = evaluate(model, dev_loader, device, threshold=config['pthresh'])
        
        print(f"\nValidation Metrics:")
        print(f"  Macro  - P: {dev_metrics['macro']['precision']:.2f}%, "
              f"R: {dev_metrics['macro']['recall']:.2f}%, "
              f"F1: {dev_metrics['macro']['f1']:.2f}%")
        print(f"  Micro  - P: {dev_metrics['micro']['precision']:.2f}%, "
              f"R: {dev_metrics['micro']['recall']:.2f}%, "
              f"F1: {dev_metrics['micro']['f1']:.2f}%")
        print(f"  Jaccard: {dev_metrics['jaccard']:.2f}%")
        
        # Update scheduler
        scheduler.step(train_loss)
        
        # Save best model
        if dev_metrics['macro']['f1'] > best_f1:
            best_f1 = dev_metrics['macro']['f1']
            best_epoch = epoch + 1
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'config': config,
                'word_to_idx': train_dataset.word_to_idx,
                'label_to_idx': train_dataset.label_to_idx
            }
            torch.save(checkpoint, data_paths['model_dump'])
            
            # Save metrics
            with open(data_paths['dev_metrics_dump'], 'w') as f:
                json.dump(dev_metrics, f, indent=2)
            
            print(f"  ✓ New best model saved! F1: {best_f1:.2f}%")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\nBest Macro-F1: {best_f1:.2f}% (Epoch {best_epoch})")
    print(f"Model saved to: {data_paths['model_dump']}")
    print(f"Metrics saved to: {data_paths['dev_metrics_dump']}")
    
    # Save vocabulary for later use
    vocab_path = 'output/vocab.json'
    with open(vocab_path, 'w') as f:
        json.dump({
            'word_to_idx': train_dataset.word_to_idx,
            'label_to_idx': train_dataset.label_to_idx
        }, f, indent=2)
    print(f"Vocabulary saved to: {vocab_path}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()