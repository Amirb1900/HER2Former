import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs.config import Config
from her2_data.dataloader import create_dataloaders

from Experiments.ex5_adaptive_diversity.her2former_ex5_adaptive import (
    HER2FormerEx5AdaptiveDiversity
)

from losses.ordinal_loss import CoralOrdinalLoss
from trainers.trainer import Trainer


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
        "HER2Former - Experiment 5: "
        "Adaptive Diversity Routing"
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

    model = HER2FormerEx5AdaptiveDiversity(
        num_classes=config.dataset.num_classes,
        top_k=64,
        freeze_backbone=config.model.freeze_backbone,
        embed_dim=config.model.embedding_dim,

        # Adaptive diversity range.
        lambda_min=0.05,
        lambda_max=0.50,

        temperature=0.5,
    )

    print("Experiment 5 Model Created.")
    print()

    print("Top-K Token Routing : ADAPTIVE DIVERSITY-AWARE")
    print("Patch Tokens        : 256")
    print("Selected Tokens     : 64")
    print("Cross Attention     : DISABLED")
    print("Diversity Weight    : IMAGE-ADAPTIVE")
    print("Lambda Min          : 0.05")
    print("Lambda Max          : 0.50")
    print("Loss                : CORAL")
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
        weight_decay=config.training.weight_decay,
    )

    # ======================================================
    # Scheduler
    # ======================================================

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.training.epochs,
    )

    # ======================================================
    # Output Directory
    # ======================================================

    save_dir = (
        "/kaggle/working/HER2Former/"
        "outputs/experiments/"
        "ex5_adaptive_diversity"
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
    # Training
    # ======================================================

    print("Starting Training...")
    print()

    trainer.fit(
        num_epochs=config.training.epochs
    )

    # ======================================================
    # Summary
    # ======================================================

    print()
    print("=" * 70)
    print("Experiment 5 Training Completed.")
    print("=" * 70)

    print()

    print(
        f"Best Validation QWK : "
        f"{trainer.best_metric:.4f}"
    )

    print()

    print(
        f"Output Directory    : "
        f"{save_dir}"
    )


if __name__ == "__main__":
    main()
