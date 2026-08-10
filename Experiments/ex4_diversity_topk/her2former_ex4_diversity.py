import torch
import torch.nn as nn

from models.virchow2 import Virchow2Encoder

from Experiments.ex4_diversity_topk.topk_selector_ex4_diversity import (
    DiversityAwareTopKSelectorEx4
)

from modules.ordinal_head import OrdinalRegressionHead


class HER2FormerEx4Diversity(nn.Module):
    """
    HER2Former - Experiment 4

    Differentiable Diversity-Aware Top-K Routing.

    Architecture
    ------------

        Input Image
             |
             v
        Virchow2 Encoder
             |
             v
        256 Patch Tokens
             |
             v
        Learnable Relevance Scoring
             |
             v
        Diversity Penalty
             |
             v
        Differentiable Top-K
             |
             v
        64-token representation
             |
             +----------------+
             |                |
             v                v
          CLS Token       Selected Tokens
             |                |
             +-------+--------+
                     |
                     v
                  Fusion
                     |
                     v
             Ordinal Head
                     |
                     v
              HER2 Logits
    """

    def __init__(
        self,
        freeze_backbone=False,
        top_k=64,
        embed_dim=1280,
        num_classes=4,
        dropout=0.1,
        diversity_weight=0.25,
        temperature=0.5,
    ):
        super().__init__()

        # ==================================================
        # Backbone
        # ==================================================

        self.encoder = Virchow2Encoder(
            freeze_backbone=freeze_backbone
        )

        # ==================================================
        # Diversity-Aware Top-K
        # ==================================================

        self.token_selector = (
            DiversityAwareTopKSelectorEx4(
                embed_dim=embed_dim,
                top_k=top_k,
                diversity_weight=diversity_weight,
                temperature=temperature,
            )
        )

        # ==================================================
        # Fusion
        # ==================================================

        self.fusion_norm = nn.LayerNorm(
            embed_dim
        )

        self.fusion_projection = nn.Sequential(

            nn.Linear(
                embed_dim * 2,
                embed_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            )
        )

        # ==================================================
        # Ordinal Head
        # ==================================================

        self.ordinal_head = OrdinalRegressionHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    # ======================================================
    # Forward
    # ======================================================

    def forward(
        self,
        x,
        return_attention=False,
    ):

        # ==================================================
        # Virchow2
        # ==================================================

        cls_token, register_tokens, patch_tokens = (
            self.encoder(x)
        )

        # ==================================================
        # Diversity-Aware Routing
        # ==================================================

        (
            selected_tokens,
            pooled_tokens,
            token_scores,
            token_indices,
            selection_weights,
        ) = self.token_selector(
            patch_tokens
        )

        # ==================================================
        # Normalize
        # ==================================================

        cls_token = self.fusion_norm(
            cls_token
        )

        pooled_tokens = self.fusion_norm(
            pooled_tokens
        )

        # ==================================================
        # Global + Local Fusion
        # ==================================================

        fused_feature = torch.cat(
            [
                cls_token,
                pooled_tokens
            ],
            dim=-1
        )

        fused_feature = self.fusion_projection(
            fused_feature
        )

        # ==================================================
        # Ordinal Prediction
        # ==================================================

        logits = self.ordinal_head(
            fused_feature
        )

        # ==================================================
        # Optional outputs
        # ==================================================

        if return_attention:

            return {
                "logits": logits,
                "token_scores": token_scores,
                "token_indices": token_indices,
                "selection_weights": selection_weights,
            }

        return logits
