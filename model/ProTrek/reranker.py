import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

class CrossAttentionReranker(nn.Module):
    def __init__(self, repr_dim: int = 1024, num_heads: int = 8, dropout: float = 0.1, num_layers: int = 2):
        """
        An enhanced multi-layer Cross-Attention Reranker.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": MultiheadAttention(embed_dim=repr_dim, num_heads=num_heads, dropout=dropout, batch_first=True),
                "norm": nn.LayerNorm(repr_dim),
                "mlp": nn.Sequential(
                    nn.Linear(repr_dim, repr_dim * 2),
                    nn.GELU(),
                    nn.Linear(repr_dim * 2, repr_dim),
                    nn.Dropout(dropout)
                ),
                "norm2": nn.LayerNorm(repr_dim)
            }) for _ in range(num_layers)
        ])
        
        self.final_scoring = nn.Sequential(
            nn.Linear(repr_dim, repr_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(repr_dim // 2, 1)
        )

    def forward(self, query_tokens, protein_residues, protein_mask=None):
        """
        Args:
            query_tokens: [batch, text_len, repr_dim]
            protein_residues: [batch, prot_len, repr_dim]
            protein_mask: [batch, prot_len]
        """
        if protein_mask is not None:
            key_padding_mask = ~(protein_mask.bool())
        else:
            key_padding_mask = None

        x = query_tokens
        for layer in self.layers:
            # Cross attention: Text (Q) attends to Protein (K, V)
            attn_output, _ = layer["cross_attn"](query=x, 
                                               key=protein_residues, 
                                               value=protein_residues,
                                               key_padding_mask=key_padding_mask)
            x = layer["norm"](x + attn_output)
            
            # Feed-forward
            mlp_output = layer["mlp"](x)
            x = layer["norm2"](x + mlp_output)
        
        # Max pool over text length
        x = x.max(dim=1)[0]
        
        # Final scoring
        score = self.final_scoring(x).squeeze(-1)
        return score

