import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Cross Attention Fusion Module

    Query:
        Global representation (CLS token)

    Key/Value:
        Selected HER2-aware patch tokens

    Input:
        cls_token:
            [B, 1280]

        selected_tokens:
            [B, K, 1280]

    Output:
        fused_feature:
            [B, 1280]
    """

    def __init__(
        self,
        embed_dim=1280,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim


        self.query_norm = nn.LayerNorm(embed_dim)
        self.token_norm = nn.LayerNorm(embed_dim)


        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )


        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),

            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )


    def forward(
        self,
        cls_token,
        selected_tokens
    ):

        """
        cls_token:
            [B,1280]

        selected_tokens:
            [B,K,1280]
        """

        # convert CLS token into query sequence

        q = cls_token.unsqueeze(1)
        # [B,1,1280]


        q = self.query_norm(q)

        kv = self.token_norm(selected_tokens)


        attn_output, attn_weights = self.cross_attention(
            query=q,
            key=kv,
            value=kv
        )


        # residual connection

        fused = q + attn_output


        # FFN

        fused = fused + self.ffn(fused)


        # remove sequence dimension

        fused = fused.squeeze(1)


        return fused, attn_weights
