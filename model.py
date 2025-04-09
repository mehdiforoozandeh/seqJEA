# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import AutoModel

# VOCAB_SIZE = 4096

# ########################################################

# def get_alibi_slopes(n):
#     """
#     Compute ALiBi slopes for n attention heads.
#     Here we use a simplified approach: for head i, we assign a slope of:
#       slope_i = 1 / (2^(i+1))
#     In practice, the ALiBi paper uses a slightly more sophisticated method.
#     """
#     return torch.tensor([1 / (2 ** (i + 1)) for i in range(n)])

# class ALiBiMultiheadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout=0.0, max_len=512):
#         """
#         Multi-head attention module that incorporates ALiBi.
#         Instead of learned or relative positional embeddings, we add a linear bias
#         based on token distance. This bias is fixed and computed using per-head slopes.
#         """
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
#         self.d_k = embed_dim // num_heads
#         self.dropout = dropout
#         self.max_len = max_len

#         # Linear projections for Q, K, V and output projection.
#         self.W_q = nn.Linear(embed_dim, embed_dim)
#         self.W_k = nn.Linear(embed_dim, embed_dim)
#         self.W_v = nn.Linear(embed_dim, embed_dim)
#         self.W_o = nn.Linear(embed_dim, embed_dim)
        
#         # Precompute slopes for each head.
#         self.register_buffer("alibi_slopes", get_alibi_slopes(num_heads).unsqueeze(1).unsqueeze(1))
#         # Shape: [num_heads, 1, 1]

#     def forward(self, x, key_padding_mask=None):
#         """
#         x: [batch, seq_len, embed_dim]
#         key_padding_mask: [batch, seq_len] with True for positions to ignore.
#         Returns: [batch, seq_len, embed_dim]
#         """
#         bsz, seq_len, _ = x.size()

#         # Compute Q, K, V and reshape to [batch, num_heads, seq_len, d_k]
#         Q = self.W_q(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         K = self.W_k(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         V = self.W_v(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)

#         # Content-based attention scores: [bsz, num_heads, seq_len, seq_len]
#         scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, H, L, L]
        
#         # Compute relative positions: [seq_len, seq_len]
#         positions = torch.arange(seq_len, device=x.device)
#         rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)  # shape [L, L]
#         # For ALiBi, we add a negative bias proportional to the distance.
#         # Expand to [1, 1, L, L] then multiply by per-head slopes.
#         alibi_bias = -self.alibi_slopes * rel_pos.to(x.dtype)  # [H, 1, L, L]
#         alibi_bias = alibi_bias.unsqueeze(0).expand(bsz, -1, -1, -1)  # [B, H, L, L]
        
#         # Add the ALiBi bias to the scores.
#         scores = scores / math.sqrt(self.d_k) + alibi_bias

#         # Apply key padding mask if provided.
#         if key_padding_mask is not None:
#             mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
#             scores = scores.masked_fill(mask, float('-inf'))

#         attn = F.softmax(scores, dim=-1)
#         attn = F.dropout(attn, p=self.dropout, training=self.training)
        
#         # Compute attention output.
#         out = torch.matmul(attn, V)  # [B, H, L, d_k]
#         out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
#         out = self.W_o(out)
#         return out

# class ALiBiTransformerEncoderLayer(nn.Module):
#     def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1, max_len=512):
#         super().__init__()
#         self.self_attn = ALiBiMultiheadAttention(embed_dim, num_heads, dropout=dropout, max_len=max_len)
#         self.linear1 = nn.Linear(embed_dim, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
        
#     def forward(self, src, src_key_padding_mask=None):
#         # Self-attention block with residual connection and layer norm.
#         src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask)
#         src = self.norm1(src + self.dropout1(src2))
#         # Feed-forward block with residual connection and layer norm.
#         src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
#         src = self.norm2(src + self.dropout2(src2))
#         return src

# class DNATransformer_ALiBi(nn.Module):
#     def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=256, num_layers=4, num_heads=4, 
#                  dim_feedforward=512, max_len=512, projection_dim=256, dropout=0.1):
#         """
#         A Transformer encoder for DNA sequences using ALiBi for positional encoding.
#         It incorporates masking for [PAD] (id=3) and [MASK] (id=4) tokens.
#         """
#         super().__init__()
#         self.embed_dim = embed_dim
        
#         # Token embedding.
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         self.max_len = max_len
        
#         # Stack of ALiBi-based transformer encoder layers.
#         self.layers = nn.ModuleList([
#             ALiBiTransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout, max_len)
#             for _ in range(num_layers)
#         ])
        
#         # Projection head (MLP).
#         self.projection_head = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.GELU(),
#             nn.Linear(embed_dim, projection_dim)
#         )
        
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x):
#         """
#         x: [batch_size, seq_len] of token IDs.
#         Returns a tuple ([batch_size, projection_dim], [batch_size, projection_dim]):
#             - cls_proj: Projection of the CLS token embedding.
#             - pooled_proj: Projection of the mean-pooled embedding (excluding masked/padded tokens).
#         Ignores [PAD] (id=3) and [MASK] (id=4) tokens during attention and pooling.
#         """
#         bsz, seq_len = x.shape
        
#         # Generate attention mask: True for [PAD] or [MASK] tokens to ignore them
#         key_padding_mask = (x == 3) | (x == 4) | (x == 2)  # [batch_size, seq_len]
        
#         # Embed tokens
#         x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
#         # Pass through transformer layers with attention mask
#         for layer in self.layers:
#             x = layer(x, src_key_padding_mask=key_padding_mask)

#         # Extract CLS token output (first token, assuming [CLS] is at position 0)
#         cls_output = x[:, 0, :]  # [batch_size, embed_dim]
        
#         # Mean pooling over sequence, excluding masked/padded positions
#         mask = (~key_padding_mask).unsqueeze(-1).float()  # [batch_size, seq_len, 1]
#         x_masked = x * mask  # Zero out masked/padded positions
#         sum_x = x_masked.sum(dim=1)  # [batch_size, embed_dim]
#         valid_counts = mask.sum(dim=1).clamp(min=1)  # [batch_size, 1]
#         pooled = sum_x / valid_counts  # [batch_size, embed_dim]
        
#         # Apply dropout
#         cls_output = self.dropout(cls_output)
#         pooled = self.dropout(pooled)
        
#         # Apply projection head to both outputs
#         cls_proj = self.projection_head(cls_output)  # [batch_size, projection_dim]
#         pooled_proj = self.projection_head(pooled)  # [batch_size, projection_dim]
        
#         return cls_proj#, pooled_proj

# ########################################################

# class RelativeMultiheadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout=0.0, max_len=512):
#         """
#         Multi-head attention with learned relative positional encodings.
#         """
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.max_len = max_len
        
#         assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
#         self.d_k = embed_dim // num_heads
        
#         # Projection layers for queries, keys, values
#         self.W_q = nn.Linear(embed_dim, embed_dim)
#         self.W_k = nn.Linear(embed_dim, embed_dim)
#         self.W_v = nn.Linear(embed_dim, embed_dim)
#         self.W_o = nn.Linear(embed_dim, embed_dim)
        
#         # Relative positional embeddings: for positions in range [-max_len+1, max_len-1]
#         self.rel_pos_embedding = nn.Embedding(2 * max_len - 1, self.d_k)
        
#     def forward(self, x, key_padding_mask=None):
#         """
#         x: [batch, seq_len, embed_dim]
#         key_padding_mask: [batch, seq_len] with True for positions to ignore.
#         Returns: [batch, seq_len, embed_dim]
#         """
#         bsz, seq_len, _ = x.size()
        
#         # Compute Q, K, V and reshape for multi-head: [batch, num_heads, seq_len, d_k]
#         Q = self.W_q(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1,2)
#         K = self.W_k(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1,2)
#         V = self.W_v(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1,2)
        
#         # Content based attention scores: [bsz, num_heads, seq_len, seq_len]
#         scores_content = torch.matmul(Q, K.transpose(-2, -1))  # Q*K^T
        
#         # Relative positional scores
#         # Create a relative position index matrix (size: seq_len x seq_len)
#         pos_ids = torch.arange(seq_len, device=x.device)
#         rel_pos = pos_ids.unsqueeze(1) - pos_ids.unsqueeze(0)  # shape [seq_len, seq_len]
#         # Shift to get indices in [0, 2*max_len-1]
#         rel_pos += self.max_len - 1
#         # Clip indices to maximum range
#         rel_pos = rel_pos.clamp(0, 2 * self.max_len - 2)
#         # Get relative positional embeddings: shape [seq_len, seq_len, d_k]
#         rel_emb = self.rel_pos_embedding(rel_pos)  # [seq_len, seq_len, d_k]
#         # Compute positional attention scores using einsum over heads: [bsz, num_heads, seq_len, seq_len]
#         scores_pos = torch.einsum('bhid,ijd->bhij', Q, rel_emb)
        
#         # Combine content and positional scores
#         scores = (scores_content + scores_pos) / math.sqrt(self.d_k)
        
#         # Apply key padding mask if provided: mask shape [bsz, seq_len]
#         if key_padding_mask is not None:
#             # Expand mask to [bsz, 1, 1, seq_len] and set masked positions to -inf
#             mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
#             scores = scores.masked_fill(mask, float('-inf'))
        
#         attn = F.softmax(scores, dim=-1)
#         attn = F.dropout(attn, p=self.dropout, training=self.training)
        
#         # Compute attention output and reshape back
#         out = torch.matmul(attn, V)  # [bsz, num_heads, seq_len, d_k]
#         out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
#         out = self.W_o(out)
#         return out

# class RelativeTransformerEncoderLayer(nn.Module):
#     def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1, max_len=512):
#         super().__init__()
#         self.self_attn = RelativeMultiheadAttention(embed_dim, num_heads, dropout=dropout, max_len=max_len)
#         self.linear1 = nn.Linear(embed_dim, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
        
#     def forward(self, src, src_key_padding_mask=None):
#         # Self-attention block with residual connection and layer norm.
#         src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask)
#         src = self.norm1(src + self.dropout1(src2))
#         # Feed-forward block with residual connection and layer norm.
#         src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
#         src = self.norm2(src + self.dropout2(src2))
#         return src

# class DNATransformer_Relative(nn.Module):
#     def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=256, num_layers=4, num_heads=4, 
#                  dim_feedforward=512, max_len=512, projection_dim=256, dropout=0.1):
#         """
#         A Transformer encoder for DNA sequences with relative positional encoding 
#         and a projection head for DINO. It also incorporates masking for masked tokens.
#         """
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.max_len = max_len
        
#         # Token embedding (learned)
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
        
#         # Instead of absolute positional embeddings, we rely on relative positional encodings in attention.
#         # However, we can still use an optional learnable embedding for absolute positions if desired.
#         # Here we omit it to emphasize relative encoding.
        
#         # Build a stack of relative transformer encoder layers.
#         self.layers = nn.ModuleList([
#             RelativeTransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout, max_len)
#             for _ in range(num_layers)
#         ])
        
#         # Projection head (MLP)
#         self.projection_head = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.GELU(),
#             nn.Linear(embed_dim, projection_dim)
#         )
        
#         # For pooling, we use mean pooling.
#         # Optionally, one could add a [CLS] token.
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         """
#         x: [batch_size, seq_len] of token IDs.
#         Returns a tuple ([batch_size, projection_dim], [batch_size, projection_dim]):
#             - cls_proj: Projection of the CLS token embedding.
#             - pooled_proj: Projection of the mean-pooled embedding (excluding masked/padded tokens).
#         Ignores [PAD] (id=3) and [MASK] (id=4) tokens during attention and pooling.
#         """
#         bsz, seq_len = x.shape
        
#         # Generate attention mask: True for [PAD] or [MASK] tokens to ignore them
#         key_padding_mask = (x == 3) | (x == 4)  # [batch_size, seq_len]
        
#         # Embed tokens
#         x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
#         # Pass through transformer layers with attention mask
#         for layer in self.layers:
#             x = layer(x, src_key_padding_mask=key_padding_mask)
        
#         # Extract CLS token output (first token, assuming [CLS] is at position 0)
#         cls_output = x[:, 0, :]  # [batch_size, embed_dim]
        
#         # Mean pooling over sequence, excluding masked/padded positions
#         mask = (~key_padding_mask).unsqueeze(-1).float()  # [batch_size, seq_len, 1]
#         x_masked = x * mask  # Zero out masked/padded positions
#         sum_x = x_masked.sum(dim=1)  # [batch_size, embed_dim]
#         valid_counts = mask.sum(dim=1).clamp(min=1)  # [batch_size, 1]
#         pooled = sum_x / valid_counts  # [batch_size, embed_dim]
        
#         # Apply dropout
#         cls_output = self.dropout(cls_output)
#         pooled = self.dropout(pooled)
        
#         # Apply projection head to both outputs
#         cls_proj = self.projection_head(cls_output)  # [batch_size, projection_dim]
#         pooled_proj = self.projection_head(pooled)  # [batch_size, projection_dim]
        
#         return cls_proj, pooled_proj

# ########################################################

# def get_sinusoid_encoding_table(n_position, d_hid):
#     """
#     Create a sinusoidal positional encoding table with shape [n_position, d_hid].
#     Based on the formula from "Attention Is All You Need".
#     """
#     def get_angle(pos, i):
#         return pos / (10000 ** (2 * (i // 2) / d_hid))
    
#     # Initialize the table
#     table = torch.zeros(n_position, d_hid)
#     for pos in range(n_position):
#         for i in range(d_hid):
#             table[pos, i] = get_angle(pos, i)
#     # Apply sin to even indices in the array; cos to odd indices.
#     table[:, 0::2] = torch.sin(table[:, 0::2])
#     table[:, 1::2] = torch.cos(table[:, 1::2])
#     return table

# class DNATransformer_Sinusoidal(nn.Module):
#     def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=256, num_layers=4, num_heads=4, 
#                  dim_feedforward=512, max_len=512, projection_dim=256, dropout=0.1):
#         """
#         A Transformer encoder for DNA sequences with sinusoidal positional encodings.
#         Incorporates masking for [PAD] (id=3) and [MASK] (id=4) tokens.
#         """
#         super().__init__()
#         self.embed_dim = embed_dim
        
#         # Token embedding layer.
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
        
#         # Precompute sinusoidal positional encodings (non-learned).
#         pe = get_sinusoid_encoding_table(max_len, embed_dim)  # shape [max_len, embed_dim]
#         self.register_buffer("pos_embedding", pe)
        
#         # Transformer encoder: using PyTorch's built-in modules.
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim, 
#             nhead=num_heads, 
#             dim_feedforward=dim_feedforward, 
#             dropout=dropout, 
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         # Projection head to obtain final DINO embedding.
#         self.projection_head = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.GELU(),
#             nn.Linear(embed_dim, projection_dim)
#         )
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         """
#         x: Tensor of shape [batch_size, seq_len] containing token IDs.
#         Returns a Tensor of shape [batch_size, projection_dim].
#         Tokens with [PAD] (3) and [MASK] (4) are ignored during attention and pooling.
#         """
#         bsz, seq_len = x.shape
        
#         # Create a key padding mask: True where token is [PAD] or [MASK].
#         key_padding_mask = (x == 3) | (x == 4)  # shape: [bsz, seq_len]
        
#         # Embed tokens.
#         token_emb = self.embedding(x)  # [bsz, seq_len, embed_dim]
        
#         # Get corresponding sinusoidal positional embeddings (slice to seq_len).
#         pos_emb = self.pos_embedding[:seq_len, :].unsqueeze(0)  # [1, seq_len, embed_dim]
        
#         # Add positional encoding to token embeddings.
#         x = token_emb + pos_emb  # [bsz, seq_len, embed_dim]
        
#         # Transformer encoder with key_padding_mask to ignore masked positions.
#         x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        
#         # Mean pooling: exclude masked positions.
#         mask = (~key_padding_mask).unsqueeze(-1).float()  # [bsz, seq_len, 1]
#         x = x * mask
#         sum_x = x.sum(dim=1)  # [bsz, embed_dim]
#         valid_counts = mask.sum(dim=1).clamp(min=1)  # [bsz, 1]
#         pooled = sum_x / valid_counts
        
#         pooled = self.dropout(pooled)
#         # Final projection to get embedding for DINO.
#         proj = self.projection_head(pooled)  # [bsz, projection_dim]
#         return proj

# ########################################################

# # Define the transformer encoder model
# class TransformerEncoder(nn.Module):
#     def __init__(self, model_name="zhihan1996/DNABERT-2-117M", hidden_size=768, context_length=512, 
#                 mask_token_id=4, pad_token_id=1, cls_token_id=0):
#         super(TransformerEncoder, self).__init__()
#         self.encoder = AutoModel.from_pretrained(model_name)
#         self.projection = nn.Linear(hidden_size, hidden_size)
#         self.context_length = context_length
#         self.mask_token_id = mask_token_id
#         self.pad_token_id = pad_token_id
#         self.cls_token_id = cls_token_id

#     def forward(self, input_ids):
#         # Dynamically generate attention mask: 1 for real tokens, 0 for mask/pad tokens
#         attention_mask = (input_ids != self.mask_token_id) & (input_ids != self.pad_token_id)
#         outputs = self.encoder(input_ids, attention_mask=attention_mask)
#         cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token output
#         avg_pool_output = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
#         return self.projection(cls_output), self.projection(avg_pool_output)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Vocabulary size constant
VOCAB_SIZE = 4096

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UnifiedDNATransformer(nn.Module):
    """
    A unified transformer model for DNA sequences that integrates four different transformer variants:
    1. DNATransformer_ALiBi (ALiBi positional encoding)
    2. DNATransformer_Relative (relative positional encoding)
    3. DNATransformer_Sinusoidal (sinusoidal positional encoding)
    4. DNABERT2 (pre-trained transformer model)
    
    The user selects the model type during initialization, and the forward pass returns the
    projection of the CLS token embedding only, matching the functionality of DNATransformer_ALiBi's
    forward method.
    """

    ### Utility Functions ###
    
    @staticmethod
    def get_alibi_slopes(n):
        """
        Compute ALiBi slopes for n attention heads using a simplified approach.
        Slope for head i = 1 / (2^(i+1)).
        
        Args:
            n (int): Number of attention heads.
        
        Returns:
            Tensor: Slopes for each head [n].
        """
        return torch.tensor([1 / (2 ** (i + 1)) for i in range(n)])

    @staticmethod
    def get_sinusoid_encoding_table(n_position, d_hid):
        """
        Create a sinusoidal positional encoding table based on "Attention Is All You Need".
        
        Args:
            n_position (int): Maximum sequence length.
            d_hid (int): Hidden dimension (embedding size).
        
        Returns:
            Tensor: Positional encoding table [n_position, d_hid].
        """
        def get_angle(pos, i):
            return pos / (10000 ** (2 * (i // 2) / d_hid))
        
        table = torch.zeros(n_position, d_hid)
        for pos in range(n_position):
            for i in range(d_hid):
                table[pos, i] = get_angle(pos, i)
        table[:, 0::2] = torch.sin(table[:, 0::2])
        table[:, 1::2] = torch.cos(table[:, 1::2])
        return table

    ### Attention and Encoder Layer Definitions ###

    class ALiBiMultiheadAttention(nn.Module):
        """Multi-head attention with ALiBi (Attention with Linear Biases) positional encoding."""
        def __init__(self, embed_dim, num_heads, dropout=0.0, max_len=512):
            """
            Initialize ALiBi multi-head attention module.
            
            Args:
                embed_dim (int): Embedding dimension.
                num_heads (int): Number of attention heads.
                dropout (float): Dropout rate.
                max_len (int): Maximum sequence length.
            """
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
            self.d_k = embed_dim // num_heads
            self.dropout = dropout
            self.max_len = max_len

            # Linear projections for queries, keys, values, and output
            self.W_q = nn.Linear(embed_dim, embed_dim)
            self.W_k = nn.Linear(embed_dim, embed_dim)
            self.W_v = nn.Linear(embed_dim, embed_dim)
            self.W_o = nn.Linear(embed_dim, embed_dim)
            
            # ALiBi slopes for positional bias
            slopes = UnifiedDNATransformer.get_alibi_slopes(num_heads)
            self.register_buffer("alibi_slopes", slopes.unsqueeze(1).unsqueeze(1))

        def forward(self, x, key_padding_mask=None):
            """
            Forward pass for ALiBi attention.
            
            Args:
                x (Tensor): Input tensor [batch, seq_len, embed_dim].
                key_padding_mask (Tensor, optional): Mask [batch, seq_len], True for positions to ignore.
            
            Returns:
                Tensor: Output tensor [batch, seq_len, embed_dim].
            """
            bsz, seq_len, _ = x.size()
            Q = self.W_q(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            
            scores = torch.matmul(Q, K.transpose(-2, -1))
            positions = torch.arange(seq_len, device=x.device)
            rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)
            alibi_bias = -self.alibi_slopes * rel_pos.to(x.dtype)
            alibi_bias = alibi_bias.unsqueeze(0).expand(bsz, -1, -1, -1)
            scores = scores / math.sqrt(self.d_k) + alibi_bias
            
            if key_padding_mask is not None:
                mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(mask, float('-inf'))
            
            attn = F.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
            return self.W_o(out)

    class ALiBiTransformerEncoderLayer(nn.Module):
        """Transformer encoder layer using ALiBi multi-head attention."""
        def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1, max_len=512):
            """
            Initialize ALiBi transformer encoder layer.
            
            Args:
                embed_dim (int): Embedding dimension.
                num_heads (int): Number of attention heads.
                dim_feedforward (int): Feedforward network dimension.
                dropout (float): Dropout rate.
                max_len (int): Maximum sequence length.
            """
            super().__init__()
            self.self_attn = UnifiedDNATransformer.ALiBiMultiheadAttention(embed_dim, num_heads, dropout, max_len)
            self.linear1 = nn.Linear(embed_dim, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, embed_dim)
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
        
        def forward(self, src, src_key_padding_mask=None):
            """
            Forward pass for ALiBi encoder layer.
            
            Args:
                src (Tensor): Input tensor [batch, seq_len, embed_dim].
                src_key_padding_mask (Tensor, optional): Mask [batch, seq_len].
            
            Returns:
                Tensor: Output tensor [batch, seq_len, embed_dim].
            """
            src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask)
            src = self.norm1(src + self.dropout1(src2))
            src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
            return self.norm2(src + self.dropout2(src2))

    class RelativeMultiheadAttention(nn.Module):
        """Multi-head attention with learned relative positional encodings."""
        def __init__(self, embed_dim, num_heads, dropout=0.0, max_len=512):
            """
            Initialize relative multi-head attention module.
            
            Args:
                embed_dim (int): Embedding dimension.
                num_heads (int): Number of attention heads.
                dropout (float): Dropout rate.
                max_len (int): Maximum sequence length.
            """
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
            self.d_k = embed_dim // num_heads
            self.dropout = dropout
            self.max_len = max_len
            
            self.W_q = nn.Linear(embed_dim, embed_dim)
            self.W_k = nn.Linear(embed_dim, embed_dim)
            self.W_v = nn.Linear(embed_dim, embed_dim)
            self.W_o = nn.Linear(embed_dim, embed_dim)
            self.rel_pos_embedding = nn.Embedding(2 * max_len - 1, self.d_k)
        
        def forward(self, x, key_padding_mask=None):
            """
            Forward pass for relative attention.
            
            Args:
                x (Tensor): Input tensor [batch, seq_len, embed_dim].
                key_padding_mask (Tensor, optional): Mask [batch, seq_len].
            
            Returns:
                Tensor: Output tensor [batch, seq_len, embed_dim].
            """
            bsz, seq_len, _ = x.size()
            Q = self.W_q(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            
            scores_content = torch.matmul(Q, K.transpose(-2, -1))
            pos_ids = torch.arange(seq_len, device=x.device)
            rel_pos = pos_ids.unsqueeze(1) - pos_ids.unsqueeze(0)
            rel_pos += self.max_len - 1
            rel_pos = rel_pos.clamp(0, 2 * self.max_len - 2)
            rel_emb = self.rel_pos_embedding(rel_pos)
            scores_pos = torch.einsum('bhid,ijd->bhij', Q, rel_emb)
            scores = (scores_content + scores_pos) / math.sqrt(self.d_k)
            
            if key_padding_mask is not None:
                mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(mask, float('-inf'))
            
            attn = F.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
            return self.W_o(out)

    class RelativeTransformerEncoderLayer(nn.Module):
        """Transformer encoder layer using relative multi-head attention."""
        def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1, max_len=512):
            """
            Initialize relative transformer encoder layer.
            
            Args:
                embed_dim (int): Embedding dimension.
                num_heads (int): Number of attention heads.
                dim_feedforward (int): Feedforward network dimension.
                dropout (float): Dropout rate.
                max_len (int): Maximum sequence length.
            """
            super().__init__()
            self.self_attn = UnifiedDNATransformer.RelativeMultiheadAttention(embed_dim, num_heads, dropout, max_len)
            self.linear1 = nn.Linear(embed_dim, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, embed_dim)
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
        
        def forward(self, src, src_key_padding_mask=None):
            """
            Forward pass for relative encoder layer.
            
            Args:
                src (Tensor): Input tensor [batch, seq_len, embed_dim].
                src_key_padding_mask (Tensor, optional): Mask [batch, seq_len].
            
            Returns:
                Tensor: Output tensor [batch, seq_len, embed_dim].
            """
            src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask)
            src = self.norm1(src + self.dropout1(src2))
            src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
            return self.norm2(src + self.dropout2(src2))

    ### Main Class Initialization ###

    def __init__(self, model_type, vocab_size=VOCAB_SIZE, embed_dim=256, num_layers=4, num_heads=4,
                 dim_feedforward=512, max_len=512, projection_dim=256, dropout=0.1,
                 model_name="zhihan1996/DNABERT-2-117M"):
        """
        Initialize the UnifiedDNATransformer with the specified model type.
        
        Args:
            model_type (str): One of ['alibi', 'relative', 'sinusoidal', 'dnabert2'].
            vocab_size (int): Vocabulary size for custom models (ignored for 'dnabert2').
            embed_dim (int): Embedding dimension for custom models.
            num_layers (int): Number of transformer layers for custom models.
            num_heads (int): Number of attention heads for custom models.
            dim_feedforward (int): Feedforward network dimension for custom models.
            max_len (int): Maximum sequence length.
            projection_dim (int): Output dimension of the projection head.
            dropout (float): Dropout rate.
            model_name (str): Pre-trained model name for 'dnabert2'.
        """
        super().__init__()
        self.model_type = model_type
        self.max_len = max_len

        if model_type in ['alibi', 'relative', 'sinusoidal']:
            # Common components for custom models
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.projection_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, projection_dim)
            )
            self.dropout = nn.Dropout(dropout)

            if model_type == 'alibi':
                # DNATransformer_ALiBi implementation
                self.layers = nn.ModuleList([
                    self.ALiBiTransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout, max_len)
                    for _ in range(num_layers)
                ])
            elif model_type == 'relative':
                # DNATransformer_Relative implementation
                self.layers = nn.ModuleList([
                    self.RelativeTransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout, max_len)
                    for _ in range(num_layers)
                ])
            elif model_type == 'sinusoidal':
                # DNATransformer_Sinusoidal implementation
                pe = self.get_sinusoid_encoding_table(max_len, embed_dim)
                self.register_buffer("pos_embedding", pe)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        elif model_type == 'dnabert2':
            # DNABERT2 (TransformerEncoder) implementation
            config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
            self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True, config=config)
            hidden_size = self.encoder.config.hidden_size
            self.projection = nn.Linear(hidden_size, projection_dim)
            # Token IDs based on DNABERT-2 defaults
            self.pad_token_id = 1
            self.mask_token_id = 4

        else:
            raise ValueError("Invalid model_type. Choose from ['alibi', 'relative', 'sinusoidal', 'dnabert2']")

    ### Forward Pass ###

    def forward(self, x):
        """
        Forward pass returning the CLS token embedding projection, consistent with DNATransformer_ALiBi.
        
        Args:
            x (Tensor): Input tensor of token IDs [batch_size, seq_len].
        
        Returns:
            Tensor: CLS token embedding projection [batch_size, projection_dim].
        """
        if self.model_type in ['alibi', 'relative']:
            bsz, seq_len = x.shape
            # Mask PAD (3) and MASK (4) tokens, but not CLS (assumed at position 0)
            key_padding_mask = (x == 3) | (x == 4)
            x = self.embedding(x)
            for layer in self.layers:
                x = layer(x, src_key_padding_mask=key_padding_mask)
            cls_output = x[:, 0, :]
            cls_output = self.dropout(cls_output)
            cls_proj = self.projection_head(cls_output)
            return cls_proj

        elif self.model_type == 'sinusoidal':
            bsz, seq_len = x.shape
            key_padding_mask = (x == 3) | (x == 4)
            token_emb = self.embedding(x)
            pos_emb = self.pos_embedding[:seq_len, :].unsqueeze(0)
            x = token_emb + pos_emb
            x = self.transformer(x, src_key_padding_mask=key_padding_mask)
            cls_output = x[:, 0, :]
            cls_output = self.dropout(cls_output)
            cls_proj = self.projection_head(cls_output)
            return cls_proj

        elif self.model_type == 'dnabert2':
            # Attention mask: 1 for tokens to attend to, 0 for PAD and MASK
            attention_mask = (x != self.pad_token_id) & (x != self.mask_token_id)
            outputs = self.encoder(x, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
            cls_proj = self.projection(cls_output)
            return cls_proj

if __name__ == "__main__":
    from transformers import AutoTokenizer
    import torch
    from transformers.models.bert.configuration_bert import BertConfig

    # Load the tokenizer for DNABERT-2
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    
    # Example DNA sequence
    dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
    
    # Tokenize the DNA sequence
    inputs = tokenizer(dna, return_tensors='pt')["input_ids"]
    
    # Print input details for verification
    print(f"Inputs: {inputs}")
    print(f"Input shape: {inputs.shape}")
    print(f"Max token id: {inputs.max().item()}")
    
    # Define all model types to test
    model_types = ['alibi', 'relative', 'sinusoidal', 'dnabert2']
    
    # Test each model variant
    for model_type in model_types:
        print(f"\nTesting {model_type} model")
        
        # Initialize the model with the specified type
        model = UnifiedDNATransformer(model_type=model_type).to(device)
        
        # Perform forward pass
        output = model(inputs.to(device))
        
        # Print output shape to confirm correctness
        print(f"Output shape: {output.shape}")
    