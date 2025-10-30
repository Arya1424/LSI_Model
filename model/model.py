import torch
from model.basicmodules import LstmNet, AttnNet

class HierAttnNet(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size=None, drop=0.1):
        super().__init__()
        
        if vocab_size is not None:
            self.word_embedding = torch.nn.Embedding(vocab_size, hidden_size)
            # CRITICAL FIX: Initialize embeddings with small values
            torch.nn.init.normal_(self.word_embedding.weight, mean=0.0, std=0.02)
            
            # ADD: Embedding normalization layer
            self.embedding_norm = torch.nn.LayerNorm(hidden_size)
            
            # Word-level LSTM and Attention
            self.sent_lstm = LstmNet(hidden_size)
            self.sent_attn = AttnNet(hidden_size, drop=drop)
        
        # Document-level LSTM and Attention (Sentence-level)
        self.doc_lstm = LstmNet(hidden_size)
        self.doc_attn = AttnNet(hidden_size, drop=drop)
        
    def forward(self, tokens=None, doc_inputs=None, mask=None, sent_dyn_context=None, doc_dyn_context=None): 
        if tokens is not None:
            # --- Word-level processing ---
            sent_inputs = self.word_embedding(tokens)
            
            # CRITICAL FIX: Normalize embeddings immediately after lookup
            sent_inputs = self.embedding_norm(sent_inputs)
            
            # flatten to 3-D: [B*S, W, H]
            sent_inputs = sent_inputs.view(-1, sent_inputs.size(2), sent_inputs.size(3))
            sent_mask = mask.view(-1, mask.size(2))
            
            if sent_dyn_context is not None:
                sent_dyn_context = sent_dyn_context.view(-1, sent_dyn_context.size(2))
                
            sent_hidden_all = self.sent_lstm(sent_inputs, sent_mask)
            sent_hidden = self.sent_attn(sent_hidden_all, sent_mask, dyn_context=sent_dyn_context) # [B*S, H]
            
            # Reshape back to document level: [B, S, H]
            doc_inputs = sent_hidden.view(tokens.size(0), tokens.size(1), -1)
            # New document mask: 1 if sentence is non-empty
            doc_mask = (mask.sum(dim=2) > 0).float()
        else:
            # If input is already sentence vectors (doc_inputs is [B, S, H])
            doc_mask = mask # Mask is already [B, S]
            
        # --- Sentence-level processing ---
        doc_hidden_all = self.doc_lstm(doc_inputs, doc_mask)
        doc_hidden = self.doc_attn(doc_hidden_all, doc_mask, dyn_context=doc_dyn_context) # [B, H]
        return doc_hidden

class MetapathAggrNet(torch.nn.Module):
    def __init__(self, node_vocab_size, edge_vocab_size, hidden_size, drop=0.1, gdel=14.):
        super().__init__()
        # CHANGED: Increase initialization range
        self.emb_range = (6.0 / hidden_size) ** 0.5  # Xavier initialization
        
        # Embedding matrices for each node type (A, C, T, S, F)
        self.node_embedding = torch.nn.ModuleDict({ntype: torch.nn.Embedding(num_nodes, hidden_size) for ntype, num_nodes in node_vocab_size.items()})
        for ntype, ntype_weights in self.node_embedding.items():
            # CHANGED: Use normal distribution instead of uniform
            torch.nn.init.normal_(ntype_weights.weight, mean=0.0, std=self.emb_range)
        
        # Linear layers to scale/transform embeddings to a common space
        self.scale_fc = torch.nn.ModuleDict({ntype: torch.nn.Linear(hidden_size, hidden_size) for ntype in node_vocab_size})
        # CHANGED: Initialize scale layers properly
        for ntype, layer in self.scale_fc.items():
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        
        # Embedding matrix for edge types (relations)
        self.edge_embedding = torch.nn.Embedding(edge_vocab_size, hidden_size // 2)
        torch.nn.init.normal_(self.edge_embedding.weight, mean=0.0, std=self.emb_range)
        
        # Intra-metapath attention (Aggregates paths/samples of one schema)
        self.intra_attention = AttnNet(2 * hidden_size, drop=drop)
        
        # Inter-metapath attention components (Aggregates across different schemas)
        self.inter_fc = torch.nn.Linear(2 * hidden_size, 2 * hidden_size)
        torch.nn.init.xavier_uniform_(self.inter_fc.weight)
        
        self.inter_context = torch.nn.Parameter(torch.rand(2 * hidden_size)) # Static context vector for inter-aggregation
        torch.nn.init.normal_(self.inter_context, mean=0.0, std=0.02)
        
        self.output_fc = torch.nn.Linear(2 * hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.output_fc.weight)
        
        # ADD: Layer normalization for output
        self.output_norm = torch.nn.LayerNorm(hidden_size)
        
        self.dropout = torch.nn.Dropout(drop)
    
    def embed_and_scale(self, tokens, edge_tokens, schema): # tokens: [B, L+1], edge_tokens: [B, L]
        inputs, edge_inputs = [], []
        
        # Embed the starting node
        node_type = schema[0][0]  
        node_input = self.dropout(self.node_embedding[node_type](tokens[:, 0])) # [B, H]
        inputs.append(self.dropout(self.scale_fc[node_type](node_input)))
        
        # Embed and scale intermediate/end nodes and edges
        for i in range(edge_tokens.size(1)):
            node_type = schema[i][2]
            node_input = self.dropout(self.node_embedding[node_type](tokens[:, i+1])) # [B, H]
            inputs.append(self.dropout(self.scale_fc[node_type](node_input)))
                          
            edge_inputs.append(self.dropout(self.edge_embedding(edge_tokens[:, i])))
            
        inputs = torch.stack(inputs, dim=1) # [B, L+1, H]
        edge_inputs = torch.stack(edge_inputs, dim=1) # [B, L, H/2]
        return inputs, edge_inputs
    
    # RotatE algorithm for relational encoding (used for metapath instance representation)
    def rotational_encoding(self, inputs, edge_inputs): # inputs: [B, L+1, H], edge_inputs: [B, L, H/2]
        PI = 3.14159265358979323846
        hidden = inputs.clone()
        
        # Iterate backwards along the path
        for i in reversed(range(edge_inputs.size(1))):
            # Split embeddings into real and imaginary parts for RotatE/complex domain
            hid_real, hid_imag = torch.chunk(hidden.clone()[:, i+1:, :], 2, dim=2) 
            inp_real, inp_imag = torch.chunk(inputs[:, i, :], 2, dim=1)
            
            # Relations are treated as rotation angles
            edge_real, edge_imag = torch.cos(edge_inputs[:, i, :]), torch.sin(edge_inputs[:, i, :])
           
            # Complex addition and rotation (h_i = h_{i-1} + r_i * h_i)
            out_real = inp_real.unsqueeze(1) + edge_real.unsqueeze(1) * hid_real - edge_imag.unsqueeze(1) * hid_imag
            out_imag = inp_imag.unsqueeze(1) + edge_imag.unsqueeze(1) * hid_real + edge_real.unsqueeze(1) * hid_imag
            
            hidden[:, i+1:, :] = torch.cat([out_real, out_imag], dim=2)
            
        path_lens = 1 + torch.arange(hidden.size(1), device=hidden.device) # [L+1]
        return hidden / path_lens.unsqueeze(0).unsqueeze(2) # Average by path length (normalization)
                               
    def forward(self, tokens, edge_tokens, schemas, intra_context=None, inter_context=None): 
        # tokens: N * [M, D, L+1], edge_tokens: N * [M, D, L] (N=num_schemas, M=num_samples, D=batch_size)
        hidden = []

        # 1. Intra-Metapath Aggregation (MAGNN - Metapath Instance Encoder + Attention)
        for i in range(len(tokens)):
            # Flatten samples: [M*D, L+1]
            mpath_tokens = tokens[i].view(-1, tokens[i].size(2)) 
            mpath_edge_tokens = edge_tokens[i].view(-1, edge_tokens[i].size(2)) 
                               
            mpath_inputs, mpath_edge_inputs = self.embed_and_scale(mpath_tokens, mpath_edge_tokens, schemas[i])
                               
            mpath_hidden_all = self.rotational_encoding(mpath_inputs, mpath_edge_inputs) # [M*D, L+1, H]

            # Concatenate target node (first element) with path/neighbor embeddings (rest)
            mpath_hidden_all = torch.cat([mpath_hidden_all[:, 0, :].unsqueeze(1).repeat(1, mpath_hidden_all.size(1) - 1, 1), mpath_hidden_all[:, 1:, :]], dim=2) # [M*D, L, 2H]                   
            
            # Apply attention over the path/samples
            mpath_hidden = torch.relu(self.intra_attention(mpath_hidden_all, dyn_context=intra_context)) # [M*D, 2H]

            # Aggregate transformed embeddings from multiple samples of the same schema 
            mpath_hidden = torch.sum(mpath_hidden.view(tokens[i].size(0), tokens[i].size(1), -1), dim=0) # [D, 2H]
            hidden.append(mpath_hidden)
            
        hidden = torch.stack(hidden, dim=1) # [D, N, 2H] (Batch, Num_Schemas, 2H)
        
        # 2. Inter-Metapath Aggregation (Attention over schemas)
        hidden_act = torch.mean(torch.tanh(self.dropout(self.inter_fc(hidden))), dim=0).expand_as(hidden) 
    
        context = self.inter_context.unsqueeze(0).repeat(hidden_act.size(0), 1).unsqueeze(2) if inter_context is None else inter_context.unsqueeze(2)
        
        scores = torch.bmm(hidden_act, context) # [D, N, 1] (Attention scores for each schema)
        
        # ADD: Use softmax to normalize attention scores
        scores = torch.nn.functional.softmax(scores.squeeze(2), dim=1).unsqueeze(2)
                            
        outputs = torch.sum(hidden * scores, dim=1) # [D, 2H] (Weighted sum across schemas)
        outputs = self.dropout(self.output_fc(outputs)) # [D, H]
        
        # ADD: Normalize output
        outputs = self.output_norm(outputs)
        
        return outputs

class MatchNet(torch.nn.Module):
    def __init__(self, hidden_size, num_labels, drop=0.1):
        super().__init__()
        
        self.match_lstm = LstmNet(hidden_size)
        self.match_attn = AttnNet(hidden_size, drop=drop)
        self.match_fc = torch.nn.Linear(2 * hidden_size, num_labels)
        
        # CRITICAL FIX: Initialize with negative bias for conservative predictions
        torch.nn.init.xavier_uniform_(self.match_fc.weight)
        # Initialize bias to -2.0 so sigmoid(logit) starts around 0.12 (below most thresholds)
        torch.nn.init.constant_(self.match_fc.bias, -2.0)
        
        self.dropout = torch.nn.Dropout(drop)
        
    def forward(self, fact_inputs, sec_inputs, context=None): # fact_inputs: [D, H], sec_inputs: [C, H]
        
        # 1. Expand Section Embeddings: [D, C, H]
        sec_inputs = sec_inputs.expand(fact_inputs.size(0), sec_inputs.size(0), sec_inputs.size(1)) 
        
        # 2. Section Contextualization (~h_s)
        sec_hidden_all = self.match_lstm(sec_inputs) # [D, C, H]
        
        # 3. Section Attention (h_S) - Dynamic context is based on fact_inputs
        sec_hidden = self.match_attn(sec_hidden_all, dyn_context=context) # [D, H]
        
        # 4. Final Classification
        logits = self.dropout(self.match_fc(torch.cat([fact_inputs, sec_hidden], dim=1))) # [D, C]
        
        # Detached scores for loss computation in helper/run.py
        scores = torch.sigmoid(logits).detach() # [D, C] 
        return logits, scores
