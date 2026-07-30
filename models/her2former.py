import torch.nn as nn

from models.virchow2 import Virchow2Encoder
from modules.lpim import LPIM
from modules.cross_attention import CrossAttention
from modules.classifier import ClassificationHead


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

        self.lpim = LPIM(
            top_k=top_k,
        )

        self.cross_attention = CrossAttention()

        self.classifier = ClassificationHead(
            num_classes=num_classes,
        )

    def forward(self, x):

        cls_token, register_tokens, patch_tokens = self.encoder(x)

        selected_tokens, indices, scores = self.lpim(
            patch_tokens,
        )

        fused_tokens, attention = self.cross_attention(
            cls_token,
            register_tokens,
            selected_tokens,
        )

        logits = self.classifier(
            fused_tokens,
        )

        return {
            "logits": logits,
            "scores": scores,
            "indices": indices,
            "attention": attention,
        }
