import torch
import torch.nn as nn


from models.virchow2 import Virchow2Encoder

from modules.topk_selector import TopKSelector
from modules.fusion import CrossAttentionFusion
from modules.ordinal_head import OrdinalRegressionHead



class HER2Former(nn.Module):
    """
    HER2Former

    Architecture:

        Input Image
              |
              v
        Virchow2 Encoder
              |
        -----------------
        |               |
        CLS Token    Patch Tokens
        [B,1280]     [B,256,1280]
                        |
                        v
                 Top-K Token Routing
                        |
                        v
              Selected Tokens
                  [B,K,1280]
                        |
                        v
             Cross Attention Fusion
                        |
                        v
              HER2 Representation
                  [B,1280]
                        |
                        v
             Ordinal Regression Head
                        |
                        v
                Ordinal Logits
                  [B,3]


    Output:

        Three ordinal boundaries:

        Grade > 0
        Grade > 1+
        Grade > 2+

    """


    def __init__(
        self,
        freeze_backbone=False,
        top_k=64,
        embed_dim=1280,
        num_heads=8,
        num_classes=4,
        dropout=0.1
    ):

        super().__init__()



        # ------------------------------------------------
        # Backbone
        # ------------------------------------------------

        self.encoder = Virchow2Encoder(
            freeze_backbone=freeze_backbone
        )



        # ------------------------------------------------
        # HER2-aware Token Routing
        # ------------------------------------------------

        self.token_selector = TopKSelector(
            embed_dim=embed_dim,
            top_k=top_k
        )



        # ------------------------------------------------
        # Cross Attention Fusion
        # ------------------------------------------------

        self.fusion = CrossAttentionFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )



        # ------------------------------------------------
        # Ordinal Regression
        # ------------------------------------------------

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
        Args:

            x:

                Image tensor

                [B,3,224,224]


        Returns:

            logits:

                [B,3]


            If return_attention=True:

                logits
                attention_weights
                token_scores
                token_indices

        """



        # ===========================
        # Virchow2
        # ===========================

        cls_token, register_tokens, patch_tokens = self.encoder(x)



        # ===========================
        # Top-K Token Routing
        # ===========================

        selected_tokens, token_scores, token_indices = self.token_selector(
            patch_tokens
        )



        # ===========================
        # Cross Attention Fusion
        # ===========================

        fused_feature, attention_weights = self.fusion(
            cls_token,
            selected_tokens
        )



        # ===========================
        # Ordinal Prediction
        # ===========================

        logits = self.ordinal_head(
            fused_feature
        )



        if return_attention:

            return {
                "logits": logits,
                "attention_weights": attention_weights,
                "token_scores": token_scores,
                "token_indices": token_indices
            }


        return logits
