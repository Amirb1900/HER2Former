import torch
import torch.nn as nn
import timm

from timm.layers import SwiGLUPacked


class Virchow2Encoder(nn.Module):
    """
    Virchow2 Feature Extractor

    Output:
        cls_token    : (B, 1280)
        reg_tokens   : (B, 4, 1280)
        patch_tokens : (B, 256, 1280)
    """

    def __init__(self, freeze_backbone: bool = False):
        super().__init__()

        self.backbone = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
# ==========================================================
# Gradient Checkpointing
# ==========================================================

if hasattr(self.backbone, "set_grad_checkpointing"):
    self.backbone.set_grad_checkpointing(True)
    print("Gradient Checkpointing Enabled.")
        if freeze_backbone:
            print("Freezing Virchow2 backbone...")
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):

        tokens = self.backbone(x)

        cls_token = tokens[:, 0]
        reg_tokens = tokens[:, 1:5]
        patch_tokens = tokens[:, 5:]

        return cls_token, reg_tokens, patch_tokens
