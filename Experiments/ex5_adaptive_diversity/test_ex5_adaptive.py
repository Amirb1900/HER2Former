import torch

from sklearn.metrics import classification_report

from configs.config import Config
from her2_data.dataloader import create_dataloaders

from Experiments.ex5_adaptive_diversity.her2former_ex5_adaptive import (
    HER2FormerEx5AdaptiveDiversity
)

from utils.ordinal import ordinal_logits_to_labels
from utils.metrics import MetricsCalculator


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
        "HER2Former - Experiment 5: TEST"
    )
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

    model = HER2FormerEx5AdaptiveDiversity(
        num_classes=config.dataset.num_classes,
        top_k=64,
        freeze_backbone=config.model.freeze_backbone,
        embed_dim=config.model.embedding_dim,

        lambda_min=0.05,
        lambda_max=0.50,

        temperature=0.5,
    )

    model = model.to(device)

    # ======================================================
    # Checkpoint
    # ======================================================

    checkpoint_path = (
        "/kaggle/working/HER2Former/"
        "outputs/experiments/"
        "ex5_adaptive_diversity/"
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
        "Best Experiment 5 model "
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

    all_heterogeneity = []
    all_adaptive_lambda = []

    with torch.no_grad():

        for images, labels, _ in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(
                images,
                return_attention=True
            )

            logits = outputs["logits"]

            predictions = (
                ordinal_logits_to_labels(
                    logits
                )
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_heterogeneity.extend(
                outputs["heterogeneity"]
                .cpu()
                .numpy()
                .tolist()
            )

            all_adaptive_lambda.extend(
                outputs["adaptive_lambda"]
                .cpu()
                .numpy()
                .tolist()
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
    print("EXPERIMENT 5 TEST RESULTS")
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

    # ======================================================
    # Adaptive Routing Statistics
    # ======================================================

    mean_heterogeneity = (
        sum(all_heterogeneity)
        /
        len(all_heterogeneity)
    )

    mean_lambda = (
        sum(all_adaptive_lambda)
        /
        len(all_adaptive_lambda)
    )

    print()
    print("=" * 70)
    print("ADAPTIVE ROUTING STATISTICS")
    print("=" * 70)

    print(
        f"Mean Heterogeneity : "
        f"{mean_heterogeneity:.6f}"
    )

    print(
        f"Mean Adaptive λ    : "
        f"{mean_lambda:.6f}"
    )

    print(
        f"Min Adaptive λ     : "
        f"{min(all_adaptive_lambda):.6f}"
    )

    print(
        f"Max Adaptive λ     : "
        f"{max(all_adaptive_lambda):.6f}"
    )

    # ======================================================
    # Classification Report
    # ======================================================

    print()
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
                "3+",
            ],
            digits=4,
            zero_division=0,
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
    print(
        "Experiment 5 Testing Completed."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
