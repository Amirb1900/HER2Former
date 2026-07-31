import torch
import torch.nn as nn


class CoralOrdinalLoss(nn.Module):
    """
    CORAL-style Ordinal Regression Loss

    Input:
        logits:
            [B, num_classes-1]

        labels:
            [B]

            Example:
                0
                1
                2
                3

    Output:
        scalar loss
    """

    def __init__(
        self
    ):
        super().__init__()

        self.loss = nn.BCEWithLogitsLoss()


    def forward(
        self,
        logits,
        labels
    ):
        """
        Args:

            logits:
                [B,3]

            labels:
                [B]

        Returns:

            loss:
                scalar
        """


        num_classes_minus_one = logits.size(1)


        # Convert class labels to ordinal targets
        #
        # Example:
        #
        # label = 2
        #
        # target:
        # [1,1,0]

        ordinal_targets = (
            torch.arange(
                num_classes_minus_one,
                device=labels.device
            )
            .unsqueeze(0)
            <
            labels.unsqueeze(1)
        )


        ordinal_targets = ordinal_targets.float()


        loss = self.loss(
            logits,
            ordinal_targets
        )


        return loss
