import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs.config import Config
from her2_data.dataloader import create_dataloaders

from Experiments.ex1_notopk.her2former_ex1_notopk import (
    HER2FormerEx1NoTopK
)

from losses.ordinal_loss import CoralOrdinalLoss
from trainers.trainer import Trainer


def main():

    # ======================================================
    # Configuration
    # ======================================================

    config = Config()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("HER2Former - Experiment 1: No Top-K")
    print("=" * 70)

    print(f"Device : {device}")
    print()

    # ======================================================
    # DataLoader
    # ======================================================

    train_loader, val_loader, test_loader = create_dataloaders(
        config
    )

    print("DataLoaders Loaded.")
    print()

    # ======================================================
    # Model
    # ======================================================

    model = HER2FormerEx1NoTopK(
        num_classes=config.dataset.num_classes,
        freeze_backbone=config.model.freeze_backbone,
        embed_dim=config.model.embedding_dim,
    )

    print("Experiment 1 Model Created.")
    print()
    print("Top-K Token Routing : DISABLED")
    print("Patch Tokens Used   : ALL 256")
    print("Cross Attention     : ENABLED")
    print()

    # ======================================================
    # Loss
    # ======================================================

    criterion = CoralOrdinalLoss()

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
    # Experiment-specific Output Directory
    # ======================================================

    save_dir = (
        "/kaggle/working/HER2Former/"
        "outputs/experiments/ex1_notopk"
    )

    # ======================================================
    # Trainer
    # ======================================================

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        early_stop_patience=15,
    )

    # ======================================================
    # Start Training
    # ======================================================

    trainer.fit(
        num_epochs=config.training.epochs
    )

    print()
    print("=" * 70)
    print("Experiment 1 Training Completed.")
    print("=" * 70)

    print()
    print(f"Best Validation QWK : {trainer.best_metric:.4f}")
    print()
    print(f"Output Directory    : {save_dir}")


if __name__ == "__main__":
    main()
