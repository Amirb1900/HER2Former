import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """
    Cross Attention

    Query:
        CLS + Register Tokens

    Key / Value:
        Selected Patch Tokens

    Input
    -----
    cls_token        : (B, 1, 1280)
    register_tokens  : (B, 4, 1280)
    patch_tokens     : (B, K, 1280)

    Output
    ------
    fused_tokens     : (B, 5, 1280)
    attention_weights: (B, 5, K)
    """

    def __init__(
        self,
        embed_dim=1280,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        cls_token,
        register_tokens,
        patch_tokens,
    ):

        query = torch.cat(
            [cls_token, register_tokens],
            dim=1,
        )

        output, attention_weights = self.attention(
            query=query,
            key=patch_tokens,
            value=patch_tokens,
            need_weights=True,
        )

        output = self.norm(output + query)

        return output, attention_weights
