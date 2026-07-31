import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs.config import Config

from datasets.dataloader import create_dataloaders

from models.her2former import HER2Former

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

    print("=" * 60)
    print("HER2Former Training")
    print("=" * 60)
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

    model = HER2Former(
        num_classes=config.dataset.num_classes,
        top_k=config.model.top_k,
        freeze_backbone=config.model.freeze_backbone
    )

    print("Model Created.")
    print()

    # ======================================================
    # Loss
    # ======================================================

    criterion = CoralOrdinalLoss()

    # ======================================================
    # Optimizer
    # ======================================================

    optimizer = AdamW(
        model.parameters(),
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
        save_dir=config.logging.save_dir
    )

    # ======================================================
    # Start Training
    # ======================================================

    trainer.fit(
        num_epochs=config.training.epochs
    )

    print()
    print("=" * 60)
    print("Training Completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
