import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

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
                 dim_feedforward=512, max_len=512, projection_dim=256, dropout=0.1):
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

        else:
            raise ValueError("Invalid model_type. Choose from ['alibi', 'relative', 'sinusoidal']")

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

if __name__ == "__main__":
    from transformers import AutoTokenizer
    import torch

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
    model_types = ['alibi', 'relative', 'sinusoidal']
    
    # Test each model variant
    for model_type in model_types:
        print(f"\nTesting {model_type} model")
        
        # Initialize the model with the specified type
        model = UnifiedDNATransformer(model_type=model_type).to(device)
        
        # Perform forward pass
        output = model(inputs.to(device))
        
        # Print output shape to confirm correctness
        print(f"Output shape: {output.shape}")
    