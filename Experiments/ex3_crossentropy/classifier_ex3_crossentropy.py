import torch
import torch.nn as nn


class ClassificationHeadEx3CrossEntropy(nn.Module):
    """
    Classification Head for Experiment 3: Cross-Entropy

    Replaces the CORAL ordinal regression head with
    a standard 4-class classification head.

    Input:
        fused_feature:
            [B, 1280]

    Output:
        logits:
            [B, 4]

    Classes:
        0
        1+
        2+
        3+
    """

    def __init__(
        self,
        embed_dim=1280,
        num_classes=4,
        dropout=0.1,
    ):
        super().__init__()

        self.classifier = nn.Sequential(

            nn.LayerNorm(
                embed_dim
            ),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                embed_dim,
                num_classes
            )
        )

    def forward(
        self,
        x
    ):
        """
        x:
            [B, 1280]

        returns:
            [B, 4]
        """

        return self.classifier(x)
