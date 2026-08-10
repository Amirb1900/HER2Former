import torch
import torch.nn as nn

from models.virchow2 import Virchow2Encoder
from modules.topk_selector import TopKSelector
from modules.ordinal_head import OrdinalRegressionHead


class HER2FormerEx2NoCrossAttention(nn.Module):

    def __init__(
        self,
        freeze_backbone=False,
        top_k=64,
        embed_dim=1280,
        num_classes=4,
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
        # Top-K Token Routing
        # ==================================================

        self.token_selector = TopKSelector(
            embed_dim=embed_dim,
            top_k=top_k
        )

        # ==================================================
        # Simple Fusion
        # ==================================================
        #
        # Cross-Attention is intentionally removed.
        #
        # The selected patch tokens are aggregated
        # using their mean representation and then
        # combined with the CLS token.
        #

        self.fusion_norm = nn.LayerNorm(
            embed_dim
        )

        self.fusion_projection = nn.Sequential(
            nn.Linear(
                embed_dim * 2,
                embed_dim
            ),
            nn.GELU(),
            nn.Dropout(dropout)
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
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor

            Input image tensor.

            Shape:
                [B, 3, 224, 224]

        return_attention : bool

            Kept for API compatibility with the main model.

            Since Cross-Attention is removed, no attention
            weights are produced.

        Returns
        -------

        logits : torch.Tensor

            Shape:
                [B, 3]

            Corresponding to:

                Grade > 0
                Grade > 1+
                Grade > 2+

        If return_attention=True:

            Returns a dictionary containing:

                logits
                token_scores
                token_indices
        """

        # ==================================================
        # Virchow2
        # ==================================================

        cls_token, register_tokens, patch_tokens = self.encoder(x)

        # cls_token:
        # [B, 1280]

        # patch_tokens:
        # [B, 256, 1280]

        # ==================================================
        # Top-K Token Routing
        # ==================================================

        selected_tokens, token_scores, token_indices = (
            self.token_selector(
                patch_tokens
            )
        )

        # selected_tokens:
        # [B, 64, 1280]

        # ==================================================
        # Simple Token Fusion
        # ==================================================
        #
        # Aggregate the selected tokens using mean pooling.
        #

        pooled_tokens = selected_tokens.mean(
            dim=1
        )

        # [B, 1280]

        pooled_tokens = self.fusion_norm(
            pooled_tokens
        )

        cls_token = self.fusion_norm(
            cls_token
        )

        # Combine global CLS representation
        # with the selected-token representation.

        fused_feature = torch.cat(
            [
                cls_token,
                pooled_tokens
            ],
            dim=-1
        )

        # [B, 2560]

        fused_feature = self.fusion_projection(
            fused_feature
        )

        # [B, 1280]

        # ==================================================
        # Ordinal Prediction
        # ==================================================

        logits = self.ordinal_head(
            fused_feature
        )

        # ==================================================
        # Optional Output
        # ==================================================

        if return_attention:

            return {
                "logits": logits,
                "token_scores": token_scores,
                "token_indices": token_indices
            }

        return logits
