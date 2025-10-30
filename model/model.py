import torch
from model.submodules import HierAttnNet, MetapathAggrNet, MatchNet

class LeSICiN(torch.nn.Module):
    def __init__(self, hidden_size, num_labels, node_vocab_size, edge_vocab_size, vocab_size=None, label_weights=None, pthresh=0.65, lambdas=(0.5, 0.5), thetas=(3, 2, 3), drop=0.1, orwell_dim=3):
        super().__init__()
        
        self.text_encoder = HierAttnNet(hidden_size, vocab_size=vocab_size)
        self.graph_encoder = MetapathAggrNet(node_vocab_size, edge_vocab_size, hidden_size)
        self.match_network = MatchNet(hidden_size, num_labels)
        
        self.match_context_transform = torch.nn.Linear(hidden_size, hidden_size)
        self.intra_context_transform = torch.nn.Linear(hidden_size, 2 * hidden_size)
        self.inter_context_transform = torch.nn.Linear(hidden_size, 2 * hidden_size)

        if orwell_dim > 0:
            self.orwell_projection = torch.nn.Sequential(
                torch.nn.Linear(orwell_dim, hidden_size // 4),  # CHANGED: Project to smaller dim
                torch.nn.ReLU(),
                torch.nn.Dropout(drop),
                torch.nn.Linear(hidden_size // 4, hidden_size),  # ADDED: Second layer
                torch.nn.Tanh()  # ADDED: Bound output to [-1, 1]
            )
            self.orwell_scale = torch.nn.Parameter(torch.tensor(0.1))  # Learnable scale factor
        
        # Layer normalization
        self.fact_norm = torch.nn.LayerNorm(hidden_size)
        self.sec_norm = torch.nn.LayerNorm(hidden_size)
        
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=label_weights)
        
        self.pred_threshold = pthresh
        self.lambdas = lambdas 
        self.thetas = thetas 
        self.dropout = torch.nn.Dropout(drop)
        
    def calculate_losses(self, logits_list, labels):
        loss = 0
        for i, logits in enumerate(logits_list):
            if logits is not None:
                loss += self.thetas[i] * self.criterion(logits, labels)
        return loss
        
    def forward(self, fact_batch, sec_batch, pthresh=None):
        if pthresh is not None:
            self.pred_threshold = pthresh        
        
        if not fact_batch.sent_vectorized:
            fact_attr_hidden = self.text_encoder(tokens=fact_batch.tokens, mask=fact_batch.mask)
        else:
            fact_attr_hidden = self.text_encoder(doc_inputs=fact_batch.doc_inputs, mask=fact_batch.mask)

        if hasattr(fact_batch, 'has_orwell_features') and fact_batch.has_orwell_features:
            # Normalize to unit norm
            orwell_normalized = torch.nn.functional.normalize(fact_batch.orwell_features, p=2, dim=1)
            orwell_proj = self.orwell_projection(orwell_normalized)
            # CHANGED: Scale down and add (instead of direct addition)
            fact_attr_hidden = fact_attr_hidden + self.orwell_scale * orwell_proj
        
        # Normalize AFTER adding Orwell features
        fact_attr_hidden = self.fact_norm(fact_attr_hidden)
        
        if not sec_batch.sent_vectorized:
            sec_attr_hidden = self.text_encoder(tokens=sec_batch.tokens, mask=sec_batch.mask)
        else:
            sec_attr_hidden = self.text_encoder(doc_inputs=sec_batch.doc_inputs, mask=sec_batch.mask)
        
        sec_attr_hidden = self.sec_norm(sec_attr_hidden)
        
        # ... rest of code stays the same

        if hasattr(fact_batch, 'has_orwell_features') and fact_batch.has_orwell_features:
            orwell_proj = self.orwell_projection(fact_batch.orwell_features)
            fact_attr_hidden = fact_attr_hidden + orwell_proj 
        
        if not sec_batch.sent_vectorized:
            sec_attr_hidden = self.text_encoder(tokens=sec_batch.tokens, mask=sec_batch.mask)
        else:
            sec_attr_hidden = self.text_encoder(doc_inputs=sec_batch.doc_inputs, mask=sec_batch.mask)
        
        attr_match_context = self.dropout(self.match_context_transform(fact_attr_hidden))
        
        attr_logits, attr_scores = self.match_network(fact_attr_hidden, sec_attr_hidden, context=attr_match_context) 
        
        sec_intra_context = self.dropout(self.intra_context_transform(sec_attr_hidden)).repeat(sec_batch.num_mpath_samples, 1)
        sec_inter_context = self.dropout(self.inter_context_transform(sec_attr_hidden))
        
        sec_struct_hidden = self.graph_encoder(sec_batch.node_tokens, sec_batch.edge_tokens, sec_batch.schemas, intra_context=sec_intra_context, inter_context=sec_inter_context) 
        
        align_logits, align_scores = self.match_network(fact_attr_hidden, sec_struct_hidden, context=attr_match_context)
        
        fact_struct_hidden = None
        if fact_batch.sample_metapaths:
            fact_intra_context = self.dropout(self.intra_context_transform(fact_attr_hidden)).repeat(fact_batch.num_mpath_samples, 1)
            fact_inter_context = self.dropout(self.inter_context_transform(fact_attr_hidden))
            
            fact_struct_hidden = self.graph_encoder(fact_batch.node_tokens, fact_batch.edge_tokens, fact_batch.schemas, intra_context=fact_intra_context, inter_context=fact_inter_context)
            
            struct_match_context = self.dropout(self.match_context_transform(fact_struct_hidden))

            struct_logits, struct_scores = self.match_network(fact_struct_hidden, sec_struct_hidden, context=struct_match_context)
            
        else:
            struct_logits = None
        
        scores = (self.lambdas[0] * attr_scores + self.lambdas[-1] * align_scores)
        predictions = (scores > self.pred_threshold).float()
        
        if fact_batch.annotated:
            loss = self.calculate_losses([attr_logits, struct_logits, align_logits], fact_batch.labels)
        else:
            loss = None
            
        return loss, predictions, sec_struct_hidden, fact_attr_hidden
