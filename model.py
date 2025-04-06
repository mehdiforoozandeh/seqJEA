import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

VOCAB_SIZE = 4096

########################################################

def get_alibi_slopes(n):
    """
    Compute ALiBi slopes for n attention heads.
    Here we use a simplified approach: for head i, we assign a slope of:
      slope_i = 1 / (2^(i+1))
    In practice, the ALiBi paper uses a slightly more sophisticated method.
    """
    return torch.tensor([1 / (2 ** (i + 1)) for i in range(n)])

class ALiBiMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, max_len=512):
        """
        Multi-head attention module that incorporates ALiBi.
        Instead of learned or relative positional embeddings, we add a linear bias
        based on token distance. This bias is fixed and computed using per-head slopes.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.d_k = embed_dim // num_heads
        self.dropout = dropout
        self.max_len = max_len

        # Linear projections for Q, K, V and output projection.
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        
        # Precompute slopes for each head.
        self.register_buffer("alibi_slopes", get_alibi_slopes(num_heads).unsqueeze(1).unsqueeze(1))
        # Shape: [num_heads, 1, 1]

    def forward(self, x, key_padding_mask=None):
        """
        x: [batch, seq_len, embed_dim]
        key_padding_mask: [batch, seq_len] with True for positions to ignore.
        Returns: [batch, seq_len, embed_dim]
        """
        bsz, seq_len, _ = x.size()

        # Compute Q, K, V and reshape to [batch, num_heads, seq_len, d_k]
        Q = self.W_q(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Content-based attention scores: [bsz, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, H, L, L]
        
        # Compute relative positions: [seq_len, seq_len]
        positions = torch.arange(seq_len, device=x.device)
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)  # shape [L, L]
        # For ALiBi, we add a negative bias proportional to the distance.
        # Expand to [1, 1, L, L] then multiply by per-head slopes.
        alibi_bias = -self.alibi_slopes * rel_pos.to(x.dtype)  # [H, 1, L, L]
        alibi_bias = alibi_bias.unsqueeze(0).expand(bsz, -1, -1, -1)  # [B, H, L, L]
        
        # Add the ALiBi bias to the scores.
        scores = scores / math.sqrt(self.d_k) + alibi_bias

        # Apply key padding mask if provided.
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        # Compute attention output.
        out = torch.matmul(attn, V)  # [B, H, L, d_k]
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        out = self.W_o(out)
        return out

class ALiBiTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1, max_len=512):
        super().__init__()
        self.self_attn = ALiBiMultiheadAttention(embed_dim, num_heads, dropout=dropout, max_len=max_len)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, src_key_padding_mask=None):
        # Self-attention block with residual connection and layer norm.
        src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        # Feed-forward block with residual connection and layer norm.
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src

class DNATransformer_ALiBi(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=256, num_layers=4, num_heads=4, 
                 dim_feedforward=512, max_len=512, projection_dim=256, dropout=0.1):
        """
        A Transformer encoder for DNA sequences using ALiBi for positional encoding.
        It incorporates masking for [PAD] (id=3) and [MASK] (id=4) tokens.
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Token embedding.
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.max_len = max_len
        
        # Stack of ALiBi-based transformer encoder layers.
        self.layers = nn.ModuleList([
            ALiBiTransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout, max_len)
            for _ in range(num_layers)
        ])
        
        # Projection head (MLP).
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, projection_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: [batch_size, seq_len] of token IDs.
        Returns a tuple ([batch_size, projection_dim], [batch_size, projection_dim]):
            - cls_proj: Projection of the CLS token embedding.
            - pooled_proj: Projection of the mean-pooled embedding (excluding masked/padded tokens).
        Ignores [PAD] (id=3) and [MASK] (id=4) tokens during attention and pooling.
        """
        bsz, seq_len = x.shape
        
        # Generate attention mask: True for [PAD] or [MASK] tokens to ignore them
        key_padding_mask = (x == 3) | (x == 4)  # [batch_size, seq_len]
        
        # Embed tokens
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # Pass through transformer layers with attention mask
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        
        # Extract CLS token output (first token, assuming [CLS] is at position 0)
        cls_output = x[:, 0, :]  # [batch_size, embed_dim]
        
        # Mean pooling over sequence, excluding masked/padded positions
        mask = (~key_padding_mask).unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        x_masked = x * mask  # Zero out masked/padded positions
        sum_x = x_masked.sum(dim=1)  # [batch_size, embed_dim]
        valid_counts = mask.sum(dim=1).clamp(min=1)  # [batch_size, 1]
        pooled = sum_x / valid_counts  # [batch_size, embed_dim]
        
        # Apply dropout
        cls_output = self.dropout(cls_output)
        pooled = self.dropout(pooled)
        
        # Apply projection head to both outputs
        cls_proj = self.projection_head(cls_output)  # [batch_size, projection_dim]
        pooled_proj = self.projection_head(pooled)  # [batch_size, projection_dim]
        
        return cls_proj, pooled_proj

########################################################

class RelativeMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, max_len=512):
        """
        Multi-head attention with learned relative positional encodings.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_len = max_len
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.d_k = embed_dim // num_heads
        
        # Projection layers for queries, keys, values
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        
        # Relative positional embeddings: for positions in range [-max_len+1, max_len-1]
        self.rel_pos_embedding = nn.Embedding(2 * max_len - 1, self.d_k)
        
    def forward(self, x, key_padding_mask=None):
        """
        x: [batch, seq_len, embed_dim]
        key_padding_mask: [batch, seq_len] with True for positions to ignore.
        Returns: [batch, seq_len, embed_dim]
        """
        bsz, seq_len, _ = x.size()
        
        # Compute Q, K, V and reshape for multi-head: [batch, num_heads, seq_len, d_k]
        Q = self.W_q(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1,2)
        K = self.W_k(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1,2)
        V = self.W_v(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1,2)
        
        # Content based attention scores: [bsz, num_heads, seq_len, seq_len]
        scores_content = torch.matmul(Q, K.transpose(-2, -1))  # Q*K^T
        
        # Relative positional scores
        # Create a relative position index matrix (size: seq_len x seq_len)
        pos_ids = torch.arange(seq_len, device=x.device)
        rel_pos = pos_ids.unsqueeze(1) - pos_ids.unsqueeze(0)  # shape [seq_len, seq_len]
        # Shift to get indices in [0, 2*max_len-1]
        rel_pos += self.max_len - 1
        # Clip indices to maximum range
        rel_pos = rel_pos.clamp(0, 2 * self.max_len - 2)
        # Get relative positional embeddings: shape [seq_len, seq_len, d_k]
        rel_emb = self.rel_pos_embedding(rel_pos)  # [seq_len, seq_len, d_k]
        # Compute positional attention scores using einsum over heads: [bsz, num_heads, seq_len, seq_len]
        scores_pos = torch.einsum('bhid,ijd->bhij', Q, rel_emb)
        
        # Combine content and positional scores
        scores = (scores_content + scores_pos) / math.sqrt(self.d_k)
        
        # Apply key padding mask if provided: mask shape [bsz, seq_len]
        if key_padding_mask is not None:
            # Expand mask to [bsz, 1, 1, seq_len] and set masked positions to -inf
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        # Compute attention output and reshape back
        out = torch.matmul(attn, V)  # [bsz, num_heads, seq_len, d_k]
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        out = self.W_o(out)
        return out

class RelativeTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1, max_len=512):
        super().__init__()
        self.self_attn = RelativeMultiheadAttention(embed_dim, num_heads, dropout=dropout, max_len=max_len)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, src_key_padding_mask=None):
        # Self-attention block with residual connection and layer norm.
        src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        # Feed-forward block with residual connection and layer norm.
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src

class DNATransformer_Relative(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=256, num_layers=4, num_heads=4, 
                 dim_feedforward=512, max_len=512, projection_dim=256, dropout=0.1):
        """
        A Transformer encoder for DNA sequences with relative positional encoding 
        and a projection head for DINO. It also incorporates masking for masked tokens.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        # Token embedding (learned)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Instead of absolute positional embeddings, we rely on relative positional encodings in attention.
        # However, we can still use an optional learnable embedding for absolute positions if desired.
        # Here we omit it to emphasize relative encoding.
        
        # Build a stack of relative transformer encoder layers.
        self.layers = nn.ModuleList([
            RelativeTransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout, max_len)
            for _ in range(num_layers)
        ])
        
        # Projection head (MLP)
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, projection_dim)
        )
        
        # For pooling, we use mean pooling.
        # Optionally, one could add a [CLS] token.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [batch_size, seq_len] of token IDs.
        Returns a tuple ([batch_size, projection_dim], [batch_size, projection_dim]):
            - cls_proj: Projection of the CLS token embedding.
            - pooled_proj: Projection of the mean-pooled embedding (excluding masked/padded tokens).
        Ignores [PAD] (id=3) and [MASK] (id=4) tokens during attention and pooling.
        """
        bsz, seq_len = x.shape
        
        # Generate attention mask: True for [PAD] or [MASK] tokens to ignore them
        key_padding_mask = (x == 3) | (x == 4)  # [batch_size, seq_len]
        
        # Embed tokens
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # Pass through transformer layers with attention mask
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        
        # Extract CLS token output (first token, assuming [CLS] is at position 0)
        cls_output = x[:, 0, :]  # [batch_size, embed_dim]
        
        # Mean pooling over sequence, excluding masked/padded positions
        mask = (~key_padding_mask).unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        x_masked = x * mask  # Zero out masked/padded positions
        sum_x = x_masked.sum(dim=1)  # [batch_size, embed_dim]
        valid_counts = mask.sum(dim=1).clamp(min=1)  # [batch_size, 1]
        pooled = sum_x / valid_counts  # [batch_size, embed_dim]
        
        # Apply dropout
        cls_output = self.dropout(cls_output)
        pooled = self.dropout(pooled)
        
        # Apply projection head to both outputs
        cls_proj = self.projection_head(cls_output)  # [batch_size, projection_dim]
        pooled_proj = self.projection_head(pooled)  # [batch_size, projection_dim]
        
        return cls_proj, pooled_proj

########################################################

def get_sinusoid_encoding_table(n_position, d_hid):
    """
    Create a sinusoidal positional encoding table with shape [n_position, d_hid].
    Based on the formula from "Attention Is All You Need".
    """
    def get_angle(pos, i):
        return pos / (10000 ** (2 * (i // 2) / d_hid))
    
    # Initialize the table
    table = torch.zeros(n_position, d_hid)
    for pos in range(n_position):
        for i in range(d_hid):
            table[pos, i] = get_angle(pos, i)
    # Apply sin to even indices in the array; cos to odd indices.
    table[:, 0::2] = torch.sin(table[:, 0::2])
    table[:, 1::2] = torch.cos(table[:, 1::2])
    return table

class DNATransformer_Sinusoidal(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=256, num_layers=4, num_heads=4, 
                 dim_feedforward=512, max_len=512, projection_dim=256, dropout=0.1):
        """
        A Transformer encoder for DNA sequences with sinusoidal positional encodings.
        Incorporates masking for [PAD] (id=3) and [MASK] (id=4) tokens.
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Token embedding layer.
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Precompute sinusoidal positional encodings (non-learned).
        pe = get_sinusoid_encoding_table(max_len, embed_dim)  # shape [max_len, embed_dim]
        self.register_buffer("pos_embedding", pe)
        
        # Transformer encoder: using PyTorch's built-in modules.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection head to obtain final DINO embedding.
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, projection_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, seq_len] containing token IDs.
        Returns a Tensor of shape [batch_size, projection_dim].
        Tokens with [PAD] (3) and [MASK] (4) are ignored during attention and pooling.
        """
        bsz, seq_len = x.shape
        
        # Create a key padding mask: True where token is [PAD] or [MASK].
        key_padding_mask = (x == 3) | (x == 4)  # shape: [bsz, seq_len]
        
        # Embed tokens.
        token_emb = self.embedding(x)  # [bsz, seq_len, embed_dim]
        
        # Get corresponding sinusoidal positional embeddings (slice to seq_len).
        pos_emb = self.pos_embedding[:seq_len, :].unsqueeze(0)  # [1, seq_len, embed_dim]
        
        # Add positional encoding to token embeddings.
        x = token_emb + pos_emb  # [bsz, seq_len, embed_dim]
        
        # Transformer encoder with key_padding_mask to ignore masked positions.
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        
        # Mean pooling: exclude masked positions.
        mask = (~key_padding_mask).unsqueeze(-1).float()  # [bsz, seq_len, 1]
        x = x * mask
        sum_x = x.sum(dim=1)  # [bsz, embed_dim]
        valid_counts = mask.sum(dim=1).clamp(min=1)  # [bsz, 1]
        pooled = sum_x / valid_counts
        
        pooled = self.dropout(pooled)
        # Final projection to get embedding for DINO.
        proj = self.projection_head(pooled)  # [bsz, projection_dim]
        return proj

########################################################

# Define the transformer encoder model
class TransformerEncoder(nn.Module):
    def __init__(self, model_name="zhihan1996/DNABERT-2-117M", hidden_size=768, context_length=512, 
                 mask_token_id=4, pad_token_id=1, cls_token_id=0):
        super(TransformerEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.context_length = context_length
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id

    def forward(self, input_ids):
        # Dynamically generate attention mask: 1 for real tokens, 0 for mask/pad tokens
        attention_mask = (input_ids != self.mask_token_id) & (input_ids != self.pad_token_id)
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token output
        avg_pool_output = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
        return self.projection(cls_output), self.projection(avg_pool_output)