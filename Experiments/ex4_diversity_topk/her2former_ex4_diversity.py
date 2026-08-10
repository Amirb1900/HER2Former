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

    Baseline:
        Experiment 2
        Virchow2
            -> Standard Top-K
            -> Representation
            -> CORAL Ordinal Head

    Experiment 4:
        Virchow2
            -> Diversity-Aware Top-K
            -> Representation
            -> CORAL Ordinal Head

    The purpose of this experiment is to determine whether
    diversity-aware token selection improves over the
    strongest current baseline (Experiment 2).

    Important:
        Cross-Attention is intentionally NOT used.

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
        Diversity-Aware Top-K
             |
             v
        64 Selected Tokens
             |
             v
        Mean Pooling
             |
             v
        HER2 Representation
             |
             v
        CORAL Ordinal Head
             |
             v
        Ordinal Logits [B,3]
    """

    def __init__(
        self,
        freeze_backbone=False,
        embed_dim=1280,
        top_k=64,
        num_classes=4,
        diversity_weight=0.25,
        dropout=0.1,
    ):
        super().__init__()

        # ==================================================
        # Virchow2 Backbone
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
        # Ordinal Regression Head
        # ==================================================

        self.ordinal_head = OrdinalRegressionHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        x,
        return_attention=False,
    ):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor

            Input image.

            Shape:
                [B, 3, 224, 224]

        return_attention : bool

            Kept for API compatibility.

            No cross-attention is used in Experiment 4.

        Returns
        -------
        logits : torch.Tensor

            Shape:
                [B, 3]

        If return_attention=True:

            returns a dictionary containing:

                logits
                selected_scores
                selected_indices
        """

        # ==================================================
        # Virchow2
        # ==================================================

        cls_token, register_tokens, patch_tokens = self.encoder(x)

        # patch_tokens:
        # [B, 256, 1280]

        # ==================================================
        # Diversity-Aware Token Selection
        # ==================================================

        selected_tokens, selected_scores, selected_indices = (
            self.token_selector(
                patch_tokens
            )
        )

        # selected_tokens:
        # [B, 64, 1280]

        # ==================================================
        # Token Aggregation
        # ==================================================
        #
        # EXP2 does not use Cross-Attention.
        #
        # Therefore, the selected tokens are aggregated
        # directly into a single HER2 representation.
        #

        fused_feature = selected_tokens.mean(
            dim=1
        )

        # [B, 1280]

        # ==================================================
        # Ordinal Regression
        # ==================================================

        logits = self.ordinal_head(
            fused_feature
        )

        # [B, 3]

        # ==================================================
        # Optional Debug Information
        # ==================================================

        if return_attention:

            return {
                "logits": logits,
                "selected_scores": selected_scores,
                "selected_indices": selected_indices,
            }

        return logits
