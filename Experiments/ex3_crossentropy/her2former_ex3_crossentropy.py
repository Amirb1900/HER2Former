import torch
import torch.nn as nn

from models.virchow2 import Virchow2Encoder

from modules.topk_selector import TopKSelector
from modules.fusion import CrossAttentionFusion

from Experiments.ex3_crossentropy.classifier_ex3_crossentropy import (
    ClassificationHeadEx3CrossEntropy
)


class HER2FormerEx3CrossEntropy(nn.Module):
    """
    HER2Former - Experiment 3: Cross-Entropy

    Ablation experiment:
        Replaces the ordinal CORAL regression head
        with a standard 4-class classification head
        trained using Cross-Entropy Loss.

    Architecture
    ------------

        Input Image
             |
             v
        Virchow2 Encoder
             |
        -------------------------
        |                       |
        CLS Token           Patch Tokens
        [B,1280]            [B,256,1280]
                                |
                                v
                       Top-K Token Routing
                                |
                                v
                        Selected Tokens
                           [B,64,1280]
                                |
                                v
                     Cross Attention Fusion
                                |
                                v
                      HER2 Representation
                           [B,1280]
                                |
                                v
                    4-Class Classification Head
                                |
                                v
                         Class Logits
                           [B,4]


    Main HER2Former
    ----------------

        Virchow2
            ->
        Top-K Token Routing
            ->
        Cross-Attention
            ->
        CORAL Ordinal Head
            ->
        [B,3]


    Experiment 3
    ------------

        Virchow2
            ->
        Top-K Token Routing
            ->
        Cross-Attention
            ->
        Standard Classification Head
            ->
        [B,4]


    Purpose
    -------

    This experiment evaluates whether the ordinal learning
    formulation (CORAL) contributes to HER2 grading performance.

    The backbone, Top-K routing, Cross-Attention fusion,
    optimizer and training setup are kept unchanged as much
    as possible.

    The only conceptual change is:

        CORAL Ordinal Regression
                    ↓
        Standard 4-Class Classification

    Loss:

        CrossEntropyLoss
    """

    def __init__(
        self,
        freeze_backbone=False,
        top_k=64,
        embed_dim=1280,
        num_heads=8,
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
        # Cross Attention Fusion
        # ==================================================

        self.fusion = CrossAttentionFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # ==================================================
        # Standard Classification Head
        # ==================================================

        self.classifier = ClassificationHeadEx3CrossEntropy(
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

        Input:
            x:
                [B, 3, 224, 224]

        Output:

            logits:
                [B, 4]

            corresponding to:

                class 0
                class 1+
                class 2+
                class 3+

        If return_attention=True:

            returns a dictionary containing:

                logits
                attention_weights
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
        # Cross Attention Fusion
        # ==================================================

        fused_feature, attention_weights = self.fusion(
            cls_token,
            selected_tokens
        )

        # fused_feature:
        # [B, 1280]

        # ==================================================
        # Classification
        # ==================================================

        logits = self.classifier(
            fused_feature
        )

        # logits:
        # [B, 4]

        # ==================================================
        # Optional Attention Output
        # ==================================================

        if return_attention:

            return {
                "logits": logits,
                "attention_weights": attention_weights,
                "token_scores": token_scores,
                "token_indices": token_indices,
            }

        return logits
