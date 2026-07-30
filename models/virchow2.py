
import torch
import torch.nn as nn
import timm
from safetensors.torch import load_file

class Virchow2Encoder(nn.Module):
    """
    Virchow-2 Feature Extractor.

    Output:
        cls_token      : (B, 1, 1280)
        register_token : (B, 4, 1280)
        patch_tokens   : (B, 256, 1280)
    """

    def __init__(self, freeze_backbone=False):

        super().__init__()

        self.backbone = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=False,
            mlp_ratio=2.66875,      # مهم!
            num_classes=0,
            dynamic_img_size=True,
        )

        checkpoint = torch.load(
            "/root/.cache/huggingface/hub/models--paige-ai--Virchow2/model.safetensors",
        )

        self.backbone.load_state_dict(checkpoint)

        if freeze_backbone:

            for p in self.backbone.parameters():

                p.requires_grad = False

    def forward(self, x):

        tokens = self.backbone.forward_features(x)

        cls_token = tokens[:, :1]

        register_tokens = tokens[:, 1:5]

        patch_tokens = tokens[:, 5:]

        return cls_token, register_tokens, patch_tokens
