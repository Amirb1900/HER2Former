import sys
from pathlib import Path

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm

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

from torch.amp import GradScaler, autocast


# ==========================================================
# Train One Epoch
# ==========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    use_amp=True,
):

    model.train()

    running_loss = 0.0

    all_labels = []
    all_predictions = []

    progress_bar = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for images, labels, _ in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast(
            device_type="cuda",
            enabled=use_amp
        ):

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=1
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = (
        running_loss /
        len(loader)
    )

    metrics = MetricsCalculator().compute(
        all_labels,
        all_predictions
    )

    return epoch_loss, metrics


# ==========================================================
# Validation
# ==========================================================

def validate(
    model,
    loader,
    criterion,
    device,
    use_amp=True,
):

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []

    progress_bar = tqdm(
        loader,
        desc="Validation",
        leave=False
    )

    with torch.no_grad():

        for images, labels, _ in progress_bar:

            images = images.to(device)
            labels = labels.to(device)

            with autocast(
                device_type="cuda",
                enabled=use_amp
            ):

                logits = model(images)

                loss = criterion(
                    logits,
                    labels
                )

            running_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_labels.extend(
                labels.detach()
                .cpu()
                .numpy()
            )

            all_predictions.extend(
                predictions.detach()
                .cpu()
                .numpy()
            )

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

    epoch_loss = (
        running_loss /
        len(loader)
    )

    metrics = MetricsCalculator().compute(
        all_labels,
        all_predictions
    )

    return epoch_loss, metrics


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
        "HER2Former - Experiment 3: Cross-Entropy"
    )
    print("=" * 70)

    print(f"Device : {device}")
    print()

    # ======================================================
    # DataLoader
    # ======================================================

    train_loader, val_loader, test_loader = (
        create_dataloaders(config)
    )

    print("DataLoaders Loaded.")
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

    print("Experiment 3 Model Created.")
    print()
    print(
        f"Top-K Token Routing : ENABLED "
        f"(K={config.model.top_k})"
    )
    print("Cross Attention     : ENABLED")
    print("Ordinal Head        : DISABLED")
    print("Classification Head : ENABLED")
    print("Loss                : Cross-Entropy")
    print()

    # ======================================================
    # Loss
    # ======================================================

    criterion = nn.CrossEntropyLoss()

    # ======================================================
    # Optimizer
    # ======================================================

    optimizer = AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    # ======================================================
    # Scheduler
    # ======================================================

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.training.epochs
    )

    # ======================================================
    # AMP
    # ======================================================

    use_amp = torch.cuda.is_available()

    scaler = GradScaler(
        "cuda",
        enabled=use_amp
    )

    # ======================================================
    # Output Directory
    # ======================================================

    save_dir = Path(
        "/kaggle/working/HER2Former/"
        "outputs/experiments/ex3_crossentropy"
    )

    checkpoint_dir = (
        save_dir /
        "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Output Directory : {save_dir}"
    )

    print()

    # ======================================================
    # Best Model Tracking
    # ======================================================

    best_metric = -1.0
    early_stop_counter = 0
    early_stop_patience = 15

    # ======================================================
    # Training
    # ======================================================

    print("=" * 70)
    print("Starting Training...")
    print("=" * 70)
    print()

    for epoch in range(
        config.training.epochs
    ):

        print()
        print(
            f"Epoch [{epoch + 1}/"
            f"{config.training.epochs}]"
        )

        print("-" * 60)

        # --------------------------------------------------
        # Train
        # --------------------------------------------------

        train_loss, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        val_loss, val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        # --------------------------------------------------
        # Scheduler
        # --------------------------------------------------

        scheduler.step()

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------

        print(
            f"Train Loss      : "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss        : "
            f"{val_loss:.4f}"
        )

        print(
            f"Train Accuracy  : "
            f"{train_metrics['accuracy']:.4f}"
        )

        print(
            f"Val Accuracy    : "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"Val Macro F1    : "
            f"{val_metrics['macro_f1']:.4f}"
        )

        print(
            f"Val QWK         : "
            f"{val_metrics['qwk']:.4f}"
        )

        # ==================================================
        # Best Model
        # ==================================================

        current_metric = val_metrics["qwk"]

        if current_metric > best_metric:

            best_metric = current_metric

            early_stop_counter = 0

            print(
                f"Best Metric : "
                f"{best_metric:.4f}"
            )

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "epoch":
                    epoch + 1,

                "best_metric":
                    best_metric,
            }

            torch.save(
                checkpoint,
                checkpoint_dir /
                "best_model.pth"
            )

            print(
                "Best model updated."
            )

        else:

            early_stop_counter += 1

            print(
                f"No improvement "
                f"({early_stop_counter}/"
                f"{early_stop_patience})"
            )

        # ==================================================
        # Last Checkpoint
        # ==================================================

        checkpoint = {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "epoch":
                epoch + 1,

            "best_metric":
                best_metric,
        }

        torch.save(
            checkpoint,
            checkpoint_dir /
            "last_checkpoint.pth"
        )

        print(
            "Checkpoint saved: "
            f"{checkpoint_dir}/last_checkpoint.pth"
        )

        # ==================================================
        # Early Stopping
        # ==================================================

        if early_stop_counter >= early_stop_patience:

            print()
            print(
                "Early stopping triggered."
            )

            break

    # ======================================================
    # Finished
    # ======================================================

    print()
    print("=" * 70)
    print("Training Finished Successfully.")
    print("=" * 70)

    print()
    print(
        f"Best Validation QWK : "
        f"{best_metric:.4f}"
    )

    print()
    print(
        f"Best Model Path:"
    )

    print(
        checkpoint_dir /
        "best_model.pth"
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()
