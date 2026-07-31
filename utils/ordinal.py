import torch


def labels_to_ordinal(labels, num_classes=4):
    """
    Convert class labels to CORAL ordinal targets.

    Example:

        label = 0 -> [0,0,0]
        label = 1 -> [1,0,0]
        label = 2 -> [1,1,0]
        label = 3 -> [1,1,1]

    Args:
        labels:
            Tensor of shape [B]

        num_classes:
            Number of classes

    Returns:
        ordinal_targets:
            Tensor of shape [B, num_classes-1]
    """

    thresholds = torch.arange(
        num_classes - 1,
        device=labels.device
    )

    ordinal_targets = (
        thresholds.unsqueeze(0)
        < labels.unsqueeze(1)
    )

    return ordinal_targets.float()


def ordinal_logits_to_probs(logits):
    """
    Convert ordinal logits into probabilities.

    Args:
        logits:
            [B, num_classes-1]

    Returns:
        probs:
            [B, num_classes-1]
    """

    return torch.sigmoid(logits)


def ordinal_logits_to_labels(logits):
    """
    Convert CORAL logits into predicted class labels.

    Rule:
        prediction = number of thresholds whose
        sigmoid(logit) > 0.5

    Example:

        [0.9,0.8,0.1] -> class 2

    Args:
        logits:
            [B, num_classes-1]

    Returns:
        labels:
            [B]
    """

    probs = ordinal_logits_to_probs(logits)

    labels = (probs > 0.5).sum(dim=1)

    return labels.long()


def ordinal_probs_to_labels(probs):
    """
    Convert probabilities into labels.

    Args:
        probs:
            [B, num_classes-1]

    Returns:
        labels:
            [B]
    """

    labels = (probs > 0.5).sum(dim=1)

    return labels.long()
