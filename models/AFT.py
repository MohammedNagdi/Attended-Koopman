import torch
import torch.nn as nn
import math

class AttentionFreeTransformer(nn.Module):
    def __init__(self, d_model, max_seq_len, device):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.max_seq_len = max_seq_len

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.position_bias = nn.Parameter(torch.randn(max_seq_len, max_seq_len))
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, d_model)  # only the last timestep
        """
        B, T, D = x.size()
        assert D == self.d_model, "Input dim mismatch"

        q = torch.sigmoid(self.Wq(x))      # (B, T, D)  
        k = self.Wk(x) / (self.d_model ** 0.5)   # (B, T, D)  division for scaling that will result in numerical stability
        v = self.Wv(x) / (self.d_model ** 0.5)   # (B, T, D)
    

        w = self.position_bias[:T, :T]     # (T, T)

        k_exp = k.unsqueeze(1)             # (B, 1, T, D)
        v_exp = v.unsqueeze(1)             # (B, 1, T, D)
        w_exp = w.unsqueeze(0).unsqueeze(-1)  # (1, T, T, 1)

        k_biased = k_exp + w_exp           # (B, T, T, D)
        exp_k = torch.exp(k_biased)        # (B, T, T, D)

        m = self.mask[:T, :T]              # (T, T)
        m_exp = m.unsqueeze(0).unsqueeze(-1)  # (1, T, T, 1)
        exp_k = exp_k.masked_fill(m_exp, 0.0)

        num = (exp_k * v_exp).sum(dim=2)   # (B, T, D)
        denom = exp_k.sum(dim=2)           # (B, T, D)
        y = q * (num / (denom + 1e-8))     # (B, T, D)

        return y[:, -1, :]                 # return only last timestep




##############################################
# Multi-Head Attention
##############################################

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, device):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.device = device
        self.max_seq_len = max_seq_len
        self.d_k = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Multi-head attention layers
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model)
        
        # Create sinusoidal positional encodings
        self.register_buffer('positional_encoding', self._create_positional_encoding(max_seq_len, d_model))
        
    def _create_positional_encoding(self, max_seq_len, d_model):
        """Create sinusoidal positional encodings as in the original Transformer paper"""
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])  # Handle odd d_model
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            (batch, d_model) - only the last timestep
        """
        B, T, D = x.size()
        assert D == self.d_model, f"Input dim {D} != expected {self.d_model}"
        
        # Add positional encodings
        pos_enc = self.positional_encoding[:T, :]  # (T, D)
        x_pos = x + pos_enc.unsqueeze(0)  # (B, T, D)
        
        # Compute Q, K, V
        Q = self.Wq(x_pos)  # (B, T, D)
        K = self.Wk(x_pos)  # (B, T, D)
        V = self.Wv(x_pos)  # (B, T, D)
        
        # Reshape for multi-head attention
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, T, d_k)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, T, d_k)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, T, d_k)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, n_heads, T, T)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)  # (B, n_heads, T, T)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # (B, n_heads, T, d_k)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        
        # Final output projection
        output = self.Wo(attended)  # (B, T, D)
        
        # Return only the last timestep
        return output[:, -1, :]  # (B, D)