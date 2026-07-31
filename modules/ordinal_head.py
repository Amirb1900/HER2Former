import torch
import torch.nn as nn


class OrdinalRegressionHead(nn.Module):
    """
    CORAL-style Ordinal Regression Head

    Input:
        fused_feature:
            [B, embed_dim]

    Output:
        ordinal_logits:
            [B, num_classes-1]

    For HER2 grading:

        Classes:
            0
            1+
            2+
            3+

        Output:
            3 ordinal boundaries:

            grade > 0
            grade > 1+
            grade > 2+
    """

    def __init__(
        self,
        embed_dim=1280,
        num_classes=4,
        dropout=0.1
    ):
        super().__init__()

        self.num_classes = num_classes


        self.norm = nn.LayerNorm(embed_dim)


        self.dropout = nn.Dropout(dropout)


        # CORAL requires K-1 binary ordinal outputs
        self.fc = nn.Linear(
            embed_dim,
            num_classes - 1
        )


    def forward(
        self,
        x
    ):
        """
        Args:
            x:
                [B,1280]

        Returns:
            logits:
                [B,3]
        """

        x = self.norm(x)

        x = self.dropout(x)

        logits = self.fc(x)


        return logits
