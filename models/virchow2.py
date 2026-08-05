import torch
import torch.nn as nn
import timm

from timm.layers import SwiGLUPacked


class Virchow2Encoder(nn.Module):
    """
    Virchow2 Feature Extractor

    Fine-tuning strategy:
        - Freeze all Virchow2 parameters
        - Train only last 4 transformer blocks

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
        # Freeze entire Virchow2 backbone
        # ==========================================================

        for param in self.backbone.parameters():
            param.requires_grad = False


        # ==========================================================
        # Unfreeze only last 4 transformer blocks
        # ==========================================================

        if hasattr(self.backbone, "blocks"):

            for block in self.backbone.blocks[-4:]:

                for param in block.parameters():
                    param.requires_grad = True

            print("Virchow2: Last 4 transformer blocks are trainable.")

        else:
            print(
                "Warning: backbone.blocks not found. "
                "Check Virchow2 architecture."
            )


        # ==========================================================
        # Gradient Checkpointing
        # ==========================================================

        if hasattr(self.backbone, "set_grad_checkpointing"):

            self.backbone.set_grad_checkpointing(True)

            print(
                "Gradient Checkpointing Enabled."
            )


        # ==========================================================
        # Trainable parameters report
        # ==========================================================

        trainable = 0
        total = 0

        for param in self.backbone.parameters():

            total += param.numel()

            if param.requires_grad:
                trainable += param.numel()


        print(
            f"Virchow2 trainable parameters: "
            f"{trainable:,} / {total:,}"
        )


    def forward(self, x):

        tokens = self.backbone(x)

        cls_token = tokens[:, 0]

        reg_tokens = tokens[:, 1:5]

        patch_tokens = tokens[:, 5:]


        return (
            cls_token,
            reg_tokens,
            patch_tokens
        )
