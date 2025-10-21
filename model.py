"""
Enhanced LeSICiN Model - FIXED VERSION
No dimension mismatch issues
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WordAttention(nn.Module):
    """Word-level attention - fixed padding handling"""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states, mask=None):
        # hidden_states: (batch, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # Compute attention scores
        scores = self.attention(hidden_states).squeeze(-1)  # (batch, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            # FIX: Ensure mask matches scores dimensions exactly
            mask_len = mask.size(1)
            if mask_len != seq_len:
                if mask_len > seq_len:
                    # Truncate mask
                    mask = mask[:, :seq_len].contiguous()
                else:
                    # Pad mask with zeros (these positions will be masked out anyway)
                    pad_size = seq_len - mask_len
                    mask = F.pad(mask, (0, pad_size), value=0)
            
            # Now mask and scores have same size
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        weights = F.softmax(scores, dim=1)  # (batch, seq_len)
        
        # Weighted sum
        attended = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        
        return attended, weights


class SentenceAttention(nn.Module):
    """Sentence-level attention - fixed padding handling"""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states, mask=None):
        # hidden_states: (batch, num_sents, hidden_size)
        batch_size, num_sents, hidden_size = hidden_states.size()
        
        # Compute attention scores
        scores = self.attention(hidden_states).squeeze(-1)  # (batch, num_sents)
        
        # Apply mask if provided
        if mask is not None:
            # FIX: Ensure mask matches scores dimensions exactly
            mask_len = mask.size(1)
            if mask_len != num_sents:
                if mask_len > num_sents:
                    # Truncate mask
                    mask = mask[:, :num_sents].contiguous()
                else:
                    # Pad mask with zeros
                    pad_size = num_sents - mask_len
                    mask = F.pad(mask, (0, pad_size), value=0)
            
            # Now mask and scores have same size
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        weights = F.softmax(scores, dim=1)  # (batch, num_sents)
        
        # Weighted sum
        attended = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        
        return attended, weights


class EnhancedLeSICiN(nn.Module):
    """
    Enhanced HAN with Orwell Features
    FIXED: No packing/unpacking to avoid dimension issues
    """
    
    def __init__(self, vocab_size, hidden_size, num_labels, orwell_dim=3, dropout=0.5):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.orwell_dim = orwell_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        
        # Word-level encoder (no dropout param to avoid issues)
        self.word_lstm = nn.LSTM(
            hidden_size, 
            hidden_size // 2, 
            bidirectional=True, 
            batch_first=True
        )
        self.word_attention = WordAttention(hidden_size)
        self.word_dropout = nn.Dropout(dropout)
        
        # Sentence-level encoder
        self.sent_lstm = nn.LSTM(
            hidden_size, 
            hidden_size // 2, 
            bidirectional=True, 
            batch_first=True
        )
        self.sent_attention = SentenceAttention(hidden_size)
        self.sent_dropout = nn.Dropout(dropout)
        
        # Orwell feature projection
        self.orwell_projection = nn.Sequential(
            nn.Linear(orwell_dim, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final classifier
        combined_size = hidden_size + hidden_size // 4
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_size, num_labels)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.embedding.weight.data[0].fill_(0)  # Padding
        
        # Initialize classifier
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, text, orwell_features):
        """
        Args:
            text: (batch_size, num_sents, num_words)
            orwell_features: (batch_size, 3)
        Returns:
            logits: (batch_size, num_labels)
        """
        batch_size, num_sents, num_words = text.size()
        
        # Create masks
        word_mask = (text != 0).float()  # (batch, num_sents, num_words)
        sent_mask = (text.sum(dim=2) != 0).float()  # (batch, num_sents)
        
        # === WORD LEVEL ===
        # Reshape for word-level processing
        text_flat = text.view(batch_size * num_sents, num_words)
        word_mask_flat = word_mask.view(batch_size * num_sents, num_words)
        
        # Embed
        embedded = self.embedding(text_flat)  # (batch*sents, words, hidden)
        embedded = self.word_dropout(embedded)
        
        # Word LSTM - NO PACKING
        word_out, _ = self.word_lstm(embedded)
        # word_out shape: (batch*sents, words, hidden)
        
        # Word attention
        sent_repr, _ = self.word_attention(word_out, word_mask_flat)
        # sent_repr shape: (batch*sents, hidden)
        
        # === SENTENCE LEVEL ===
        # Reshape back
        sent_repr = sent_repr.view(batch_size, num_sents, -1)
        sent_repr = self.sent_dropout(sent_repr)
        
        # Sentence LSTM - NO PACKING
        sent_out, _ = self.sent_lstm(sent_repr)
        # sent_out shape: (batch, num_sents, hidden)
        
        # Sentence attention
        doc_repr, _ = self.sent_attention(sent_out, sent_mask)
        # doc_repr shape: (batch, hidden)
        
        # === ORWELL FEATURES ===
        orwell_proj = self.orwell_projection(orwell_features)
        # orwell_proj shape: (batch, hidden//4)
        
        # === COMBINE ===
        combined = torch.cat([doc_repr, orwell_proj], dim=-1)
        # combined shape: (batch, hidden + hidden//4)
        
        # === CLASSIFY ===
        logits = self.classifier(combined)
        # logits shape: (batch, num_labels)
        
        return logits
    
    def get_attention_weights(self, text, orwell_features):
        """Get attention weights for visualization"""
        batch_size, num_sents, num_words = text.size()
        
        word_mask = (text != 0).float()
        sent_mask = (text.sum(dim=2) != 0).float()
        
        # Word level
        text_flat = text.view(batch_size * num_sents, num_words)
        word_mask_flat = word_mask.view(batch_size * num_sents, num_words)
        
        embedded = self.embedding(text_flat)
        word_out, _ = self.word_lstm(embedded)
        
        _, word_weights = self.word_attention(word_out, word_mask_flat)
        word_weights = word_weights.view(batch_size, num_sents, num_words)
        
        # Sentence level
        sent_repr, _ = self.word_attention(word_out, word_mask_flat)
        sent_repr = sent_repr.view(batch_size, num_sents, -1)
        
        sent_out, _ = self.sent_lstm(sent_repr)
        _, sent_weights = self.sent_attention(sent_out, sent_mask)
        
        return word_weights, sent_weights


if __name__ == '__main__':
    print("Testing Enhanced LeSICiN Model (Fixed)...")
    
    # Create model
    model = EnhancedLeSICiN(
        vocab_size=10000,
        hidden_size=200,
        num_labels=100,
        orwell_dim=3,
        dropout=0.5
    )
    
    # Test with various input sizes
    for batch_size in [2, 4]:
        for num_sents in [30, 50]:
            for num_words in [80, 100]:
                text = torch.randint(0, 10000, (batch_size, num_sents, num_words))
                orwell = torch.randn(batch_size, 3)
                
                logits = model(text, orwell)
                assert logits.shape == (batch_size, 100), f"Wrong shape: {logits.shape}"
    
    print("✓ All tests passed!")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")