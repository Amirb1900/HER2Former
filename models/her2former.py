import torch
import torch.nn as nn

from models.virchow2 import Virchow2Encoder


class HER2Former(nn.Module):

    def __init__(
        self,
        num_classes=4,
        top_k=64,
        freeze_backbone=False,
    ):
        super().__init__()

        self.encoder = Virchow2Encoder(
            freeze_backbone=freeze_backbone,
        )

        self.top_k = top_k

        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256,num_classes)
        )

    def forward(self,x):

        cls_token, register_tokens, patch_tokens = self.encoder(x)

        logits = self.classifier(cls_token)

        return logits
