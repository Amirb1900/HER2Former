import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    HER2 Classification Head

    Input:
        (B, 5, 1280)

    Output:
        logits (B, num_classes)
    """

    def __init__(
        self,
        embed_dim=1280,
        hidden_dim=512,
        num_classes=4,
        dropout=0.2,
    ):
        super().__init__()

        self.head = nn.Sequential(

            nn.Linear(embed_dim, hidden_dim),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_classes),

        )

    def forward(self, fused_tokens):

        cls_embedding = fused_tokens[:, 0]

        logits = self.head(cls_embedding)

        return logits
