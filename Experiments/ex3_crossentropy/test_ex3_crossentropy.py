import sys

import torch
from sklearn.metrics import classification_report

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = "/kaggle/working/HER2Former"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==========================================================
# Imports
# ==========================================================

from configs.config import Config

from her2_data.dataloader import create_dataloaders

from Experiments.ex3_crossentropy.her2former_ex3_crossentropy import (
    HER2FormerEx3CrossEntropy
)

from utils.metrics import MetricsCalculator


# ==========================================================
# Main
# ==========================================================

def main():

    # ======================================================
    # Configuration
    # ======================================================

    config = Config()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print(
        "HER2Former - Experiment 3: "
        "Cross-Entropy TEST"
    )
    print("=" * 70)

    print(
        f"Device : {device}"
    )

    print()

    # ======================================================
    # DataLoader
    # ======================================================

    _, _, test_loader = create_dataloaders(
        config
    )

    print(
        "Test DataLoader Loaded."
    )

    print()

    # ======================================================
    # Model
    # ======================================================

    model = HER2FormerEx3CrossEntropy(
        num_classes=config.dataset.num_classes,
        top_k=config.model.top_k,
        freeze_backbone=config.model.freeze_backbone,
        embed_dim=config.model.embedding_dim,
    )

    model = model.to(device)

    print(
        "Experiment 3 model created."
    )

    print(
        f"Top-K : {config.model.top_k}"
    )

    print(
        "Cross Attention : ENABLED"
    )

    print(
        "Classification Head : ENABLED"
    )

    print()

    # ======================================================
    # Checkpoint
    # ======================================================

    checkpoint_path = (
        "/kaggle/working/HER2Former/"
        "outputs/experiments/"
        "ex3_crossentropy/"
        "checkpoints/"
        "best_model.pth"
    )

    print(
        "Loading checkpoint:"
    )

    print(
        checkpoint_path
    )

    print()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Best Experiment 3 model "
        "loaded successfully."
    )

    print()

    print(
        f"Best Validation QWK : "
        f"{checkpoint['best_metric']:.4f}"
    )

    print(
        f"Checkpoint Epoch    : "
        f"{checkpoint['epoch']}"
    )

    print()

    # ======================================================
    # Evaluation
    # ======================================================

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels, _ in test_loader:

            images = images.to(device)

            labels = labels.to(device)

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            logits = model(
                images
            )

            # ------------------------------------------------
            # Standard Classification
            # ------------------------------------------------

            predictions = torch.argmax(
                logits,
                dim=1
            )

            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    # ======================================================
    # Metrics
    # ======================================================

    metrics = MetricsCalculator().compute(
        all_labels,
        all_predictions
    )

    print()

    print("=" * 70)
    print(
        "EXPERIMENT 3 TEST RESULTS"
    )
    print("=" * 70)

    print(
        f"Accuracy     : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Acc : "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1     : "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1  : "
        f"{metrics['weighted_f1']:.4f}"
    )

    print(
        f"QWK          : "
        f"{metrics['qwk']:.4f}"
    )

    print()

    # ======================================================
    # Classification Report
    # ======================================================

    print(
        "Classification Report:"
    )

    print()

    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1, 2, 3],
            target_names=[
                "0",
                "1+",
                "2+",
                "3+"
            ],
            digits=4,
            zero_division=0
        )
    )

    # ======================================================
    # Confusion Matrix
    # ======================================================

    print(
        "Confusion Matrix:"
    )

    print()

    print(
        metrics["confusion_matrix"]
    )

    print()

    print("=" * 70)
    print(
        "Experiment 3 Testing Completed."
    )
    print("=" * 70)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()
