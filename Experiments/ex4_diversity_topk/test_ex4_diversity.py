```python
import torch
from sklearn.metrics import classification_report

from configs.config import Config
from her2_data.dataloader import create_dataloaders

from Experiments.ex4_diversity_topk.her2former_ex4_diversity import (
    HER2FormerEx4Diversity
)

from utils.ordinal import ordinal_logits_to_labels
from utils.metrics import MetricsCalculator


def main():

    # ======================================================
    # Configuration
    # ======================================================

    config = Config()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("HER2Former - Experiment 4: TEST")
    print("=" * 70)

    print(f"Device : {device}")
    print()

    # ======================================================
    # DataLoader
    # ======================================================

    _, _, test_loader = create_dataloaders(
        config
    )

    print("Test DataLoader Loaded.")
    print()

    # ======================================================
    # Model
    # ======================================================

    model = HER2FormerEx4Diversity(
        num_classes=config.dataset.num_classes,
        freeze_backbone=config.model.freeze_backbone,
        embed_dim=config.model.embedding_dim,
        top_k=64,
        diversity_weight=0.25,
    )

    model = model.to(device)

    print("Experiment 4 Model Created.")
    print()
    print("Top-K Token Routing : DIVERSITY-AWARE")
    print("Candidate Tokens    : 128")
    print("Selected Tokens     : 64")
    print("Diversity Weight    : 0.25")
    print("Cross Attention     : DISABLED")
    print()

    # ======================================================
    # Checkpoint
    # ======================================================

    checkpoint_path = (
        "/kaggle/working/HER2Former/"
        "outputs/experiments/ex4_diversity_topk/"
        "checkpoints/best_model.pth"
    )

    print("Loading checkpoint:")
    print(checkpoint_path)
    print()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Best Experiment 4 model loaded successfully."
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

            logits = model(images)

            predictions = ordinal_logits_to_labels(
                logits
            )

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
    print("EXPERIMENT 4 TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy     : {metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Acc : {metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1     : {metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1  : {metrics['weighted_f1']:.4f}"
    )

    print(
        f"QWK          : {metrics['qwk']:.4f}"
    )

    print()

    # ======================================================
    # Classification Report
    # ======================================================

    print("Classification Report:")
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

    print("Confusion Matrix:")
    print()

    print(
        metrics["confusion_matrix"]
    )

    print()

    print("=" * 70)
    print("Experiment 4 Testing Completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

