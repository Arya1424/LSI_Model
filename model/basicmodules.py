import torch

class LstmNet(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # hidden_size is always the full size (e.g., 200)
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)
        
    def forward(self, inputs, mask=None): # inputs: [B, S, H], mask: [B, S]
        mask = mask if mask is not None else torch.ones(inputs.size(0), inputs.size(1), device=inputs.device)
        lengths = mask.sum(dim=-1) # [B,] - Contains the actual sequence lengths
        
                # Filter out zero-length sequences to prevent the crash
        non_empty_mask = (lengths > 0)
        
        if not non_empty_mask.any():
            # If the entire batch is empty (all lengths are 0), return a zero tensor
            return torch.zeros_like(inputs)
            
        # 1. Filter inputs and lengths to include only non-empty sequences
        inputs_filtered = inputs[non_empty_mask]
        # .cpu() is required for pack_padded_sequence lengths argument
        lengths_filtered = lengths[non_empty_mask].cpu() 
        
        # 2. Pack, run LSTM, and unpack
        pck_inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs_filtered, lengths_filtered, batch_first=True, enforce_sorted=False)
        pck_hidden_all = self.lstm(pck_inputs)[0]
        hidden_all_filtered = torch.nn.utils.rnn.pad_packed_sequence(pck_hidden_all, batch_first=True)[0]
        
        # 3. Create output tensor of original batch size and fill it
        
        # The unpacked output's sequence length (dim 1) may vary, use its max length
        max_unpacked_len = hidden_all_filtered.size(1)
        
        # Create output tensor (original batch size, new max length, hidden size)
        hidden_all = torch.zeros(inputs.size(0), max_unpacked_len, inputs.size(2), device=inputs.device)
        
        # Fill the output tensor with the results from the non-empty sequences
        hidden_all[non_empty_mask, :max_unpacked_len, :] = hidden_all_filtered
        
        return hidden_all

class AttnNet(torch.nn.Module):
    def __init__(self, hidden_size, drop=0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.attn_fc = torch.nn.Linear(hidden_size, hidden_size)
        self.context = torch.nn.Parameter(torch.rand(hidden_size))
        
        self.dropout = torch.nn.Dropout(drop)
        
    def forward(self, inputs, mask=None, dyn_context=None): # [B, S, H], [B, S], [B, H]
        
        mask = mask if mask is not None else torch.ones(inputs.size(0), inputs.size(1), device=inputs.device)
        context = dyn_context if dyn_context is not None else self.context.expand(inputs.size(0), self.hidden_size) # [B, H]
        
        act_inputs = torch.tanh(self.dropout(self.attn_fc(inputs)))
        
        scores = torch.bmm(act_inputs, context.unsqueeze(2)).squeeze(2) # [B, S]
        msk_scores = scores.masked_fill((1 - mask).bool(), -1e-32)
        msk_scores = torch.nn.functional.softmax(msk_scores, dim=1)
        
        hidden = torch.sum(inputs * msk_scores.unsqueeze(2), dim=1) # [B, H]
        return hidden
