import torch
import torch.nn as nn

from models.virchow2 import Virchow2Encoder

from Experiments.ex4_diversity_topk.diversity_topk_ex4 import (
    DiversityAwareTopKEx4
)

from modules.ordinal_head import OrdinalRegressionHead


class HER2FormerEx4Diversity(nn.Module):
    """
    HER2Former - Experiment 4

    Diversity-Aware Top-K Token Routing

    Controlled comparison against Experiment 2.

    Experiment 2:
        Virchow2
            -> Standard Top-K
            -> Mean Pooling
            -> CLS Fusion
            -> CORAL

    Experiment 4:
        Virchow2
            -> Diversity-Aware Top-K
            -> Mean Pooling
            -> CLS Fusion
            -> CORAL

    The only architectural change is the token
    selection strategy.
    """

    def __init__(
        self,
        freeze_backbone=False,
        top_k=64,
        embed_dim=1280,
        num_classes=4,
        diversity_weight=0.25,
        dropout=0.1,
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

        self.token_selector = DiversityAwareTopKEx4(
            embed_dim=embed_dim,
            top_k=top_k,
            diversity_weight=diversity_weight,
        )

        # ==================================================
        # Simple Fusion
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
        # Ordinal Regression Head
        # ==================================================

        self.ordinal_head = OrdinalRegressionHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(
        self,
        x,
        return_attention=False
    ):

        # ==================================================
        # Virchow2
        # ==================================================

        cls_token, register_tokens, patch_tokens = (
            self.encoder(x)
        )

        # patch_tokens:
        # [B, 256, 1280]

        # ==================================================
        # Diversity-Aware Top-K
        # ==================================================

        selected_tokens, token_scores, token_indices = (
            self.token_selector(
                patch_tokens
            )
        )

        # selected_tokens:
        # [B, 64, 1280]

        # ==================================================
        # Mean Pooling
        # ==================================================

        pooled_tokens = selected_tokens.mean(
            dim=1
        )

        # [B, 1280]

        # ==================================================
        # Normalization
        # ==================================================

        pooled_tokens = self.fusion_norm(
            pooled_tokens
        )

        cls_token = self.fusion_norm(
            cls_token
        )

        # ==================================================
        # CLS + Patch Representation
        # ==================================================

        fused_feature = torch.cat(
            [
                cls_token,
                pooled_tokens
            ],
            dim=-1
        )

        # [B, 2560]

        # ==================================================
        # Projection
        # ==================================================

        fused_feature = self.fusion_projection(
            fused_feature
        )

        # [B, 1280]

        # ==================================================
        # CORAL Ordinal Head
        # ==================================================

        logits = self.ordinal_head(
            fused_feature
        )

        # [B, 3]

        # ==================================================
        # Optional Debug Output
        # ==================================================

        if return_attention:

            return {
                "logits": logits,
                "token_scores": token_scores,
                "token_indices": token_indices,
            }

        return logits
