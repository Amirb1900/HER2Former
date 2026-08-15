import torch
import torch.nn as nn

from models.virchow2 import Virchow2Encoder

from Experiments.ex5_adaptive_diversity.topk_selector_ex5_adaptive import (
    AdaptiveDiversityTopKSelectorEx5
)

from modules.ordinal_head import OrdinalRegressionHead


class HER2FormerEx5AdaptiveDiversity(nn.Module):
    """
    HER2Former - Experiment 5

    Adaptive Diversity-Aware Token Routing.

    Architecture
    ------------

        H&E Image
            |
            v
        Virchow2
            |
            v
        256 Patch Tokens
            |
            v
        HER2 Relevance Scoring
            |
            v
        Image Heterogeneity
            |
            v
        Adaptive Diversity Weight
            |
            v
        Diversity-Aware Routing
            |
            v
        Top-K = 64
            |
            +----------------+
            |                |
           CLS        Selected Patch Representation
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
        lambda_min=0.05,
        lambda_max=0.50,
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
        # Adaptive Diversity Routing
        # ==================================================

        self.token_selector = (
            AdaptiveDiversityTopKSelectorEx5(
                embed_dim=embed_dim,
                top_k=top_k,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
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
        # Adaptive Diversity Routing
        # ==================================================

        (
            selected_tokens,
            pooled_tokens,
            token_scores,
            token_indices,
            selection_weights,
            heterogeneity,
            adaptive_lambda,
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
                "heterogeneity": heterogeneity,
                "adaptive_lambda": adaptive_lambda,
            }

        return logits
