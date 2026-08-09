import torch
import torch.nn as nn


class CrossAttentionFusionEx1NoTopK(nn.Module):
    """
    Cross Attention Fusion for Experiment 1: No Top-K

    This module is intentionally equivalent to the
    CrossAttentionFusion module used in the main model,
    except that it receives ALL patch tokens directly.

    Query
    -----
    Global representation (CLS token)

        [B, 1280]

    Key / Value
    -----------
    All Virchow2 patch tokens

        [B, 256, 1280]

    Output
    ------
    fused_feature

        [B, 1280]

    Attention weights

        [B, 1, 256]

    """

    def __init__(
        self,
        embed_dim=1280,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # ==================================================
        # Normalization
        # ==================================================

        self.query_norm = nn.LayerNorm(
            embed_dim
        )

        self.token_norm = nn.LayerNorm(
            embed_dim
        )

        # ==================================================
        # Cross Attention
        # ==================================================

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ==================================================
        # Feed Forward Network
        # ==================================================

        self.ffn = nn.Sequential(

            nn.LayerNorm(
                embed_dim
            ),

            nn.Linear(
                embed_dim,
                embed_dim * 4
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                embed_dim * 4,
                embed_dim
            ),

            nn.Dropout(
                dropout
            ),
        )

    def forward(
        self,
        cls_token,
        patch_tokens,
    ):
        """
        Forward pass.

        Parameters
        ----------
        cls_token : torch.Tensor

            Shape:
                [B, 1280]

        patch_tokens : torch.Tensor

            Shape:
                [B, 256, 1280]

            IMPORTANT:
            All 256 patch tokens are used.
            No token selection is performed.

        Returns
        -------
        fused_feature : torch.Tensor

            Shape:
                [B, 1280]

        attn_weights : torch.Tensor

            Shape:
                [B, 1, 256]
        """

        # ==================================================
        # Convert CLS token into query sequence
        # ==================================================

        q = cls_token.unsqueeze(1)

        # [B, 1, 1280]

        q = self.query_norm(
            q
        )

        # ==================================================
        # Normalize all patch tokens
        # ==================================================

        kv = self.token_norm(
            patch_tokens
        )

        # [B, 256, 1280]

        # ==================================================
        # Cross Attention
        # ==================================================

        attn_output, attn_weights = self.cross_attention(
            query=q,
            key=kv,
            value=kv,
        )

        # attn_output:
        # [B, 1, 1280]

        # attn_weights:
        # [B, 1, 256]

        # ==================================================
        # Residual Connection
        # ==================================================

        fused = q + attn_output

        # ==================================================
        # Feed Forward Network
        # ==================================================

        fused = fused + self.ffn(
            fused
        )

        # ==================================================
        # Remove sequence dimension
        # ==================================================

        fused = fused.squeeze(1)

        # [B, 1280]

        return fused, attn_weights
