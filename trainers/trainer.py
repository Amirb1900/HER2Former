import os
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from utils.metrics import MetricsCalculator
from utils.ordinal import ordinal_logits_to_labels
from utils.checkpoint import CheckpointManager


class Trainer:
    """
    Generic Trainer for HER2Former.

    Responsibilities
    ----------------
    - Training loop
    - Validation loop
    - Mixed Precision Training (AMP)
    - Gradient Clipping
    - Metric Computation
    - Checkpoint Saving
    - Best Model Saving
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        save_dir="./outputs",
        max_grad_norm=1.0,
        use_amp=True,
    ):

        self.model = model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device

        self.max_grad_norm = max_grad_norm

        self.use_amp = (
            use_amp and
            torch.cuda.is_available()
        )

        self.scaler = GradScaler(
            enabled=self.use_amp
        )

        self.metrics = MetricsCalculator()

        self.save_dir = Path(save_dir)

        self.checkpoint_manager = CheckpointManager(
            self.save_dir / "checkpoints"
        )

        self.best_metric = -1.0

        self.start_epoch = 0

        os.makedirs(
            self.save_dir,
            exist_ok=True
        )

        os.makedirs(
            self.save_dir / "logs",
            exist_ok=True
        )

        os.makedirs(
            self.save_dir / "predictions",
            exist_ok=True
        )

        print("=" * 60)
        print("Trainer initialized")
        print(f"Device          : {self.device}")
        print(f"AMP             : {self.use_amp}")
        print(f"Save Directory  : {self.save_dir}")
        print(f"Start Epoch     : {self.start_epoch}")
        print(f"Best Metric     : {self.best_metric:.4f}")
        print("=" * 60)
        
    def train_one_epoch(self, epoch):
        """
        Train the model for one epoch.

        Returns
        -------
        train_loss : float
        metrics : dict
        """

        self.model.train()

        running_loss = 0.0

        all_labels = []
        all_predictions = []

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Train Epoch {epoch}",
            leave=False
        )

        for images, labels, _ in progress_bar:

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):

                logits = self.model(images)

                loss = self.criterion(
                    logits,
                    labels
                )

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

            predictions = ordinal_logits_to_labels(
                logits
            )

            all_labels.extend(
                labels.detach().cpu().numpy()
            )

            all_predictions.extend(
                predictions.detach().cpu().numpy()
            )

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        epoch_loss = (
            running_loss /
            len(self.train_loader)
        )

        metrics = self.metrics.compute(
            all_labels,
            all_predictions
        )

        return epoch_loss, metrics

    def validate(self, epoch):
        """
        Validate the model for one epoch.

        Returns
        -------
        val_loss : float
        metrics : dict
        """

        self.model.eval()

        running_loss = 0.0

        all_labels = []
        all_predictions = []

        progress_bar = tqdm(
            self.val_loader,
            desc=f"Validation {epoch}",
            leave=False
        )

        with torch.no_grad():

            for images, labels, _ in progress_bar:

                images = images.to(self.device)
                labels = labels.to(self.device)

                with autocast(enabled=self.use_amp):

                    logits = self.model(images)

                    loss = self.criterion(
                        logits,
                        labels
                    )

                running_loss += loss.item()

                predictions = ordinal_logits_to_labels(
                    logits
                )

                all_labels.extend(
                labels.detach().cpu().numpy()
                )

                all_predictions.extend(
                predictions.detach().cpu().numpy()
                )

                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

        epoch_loss = (
            running_loss /
            len(self.val_loader)
        )

        metrics = self.metrics.compute(
            all_labels,
            all_predictions
        )

        return epoch_loss, metrics

    def fit(self, num_epochs):
        """
        Main training loop.
        """

        print("\nStarting Training...\n")

        print(f"Training starts from Epoch {self.start_epoch + 1}")
        print(f"Current Best Metric : {self.best_metric:.4f}")
        print()

        for epoch in range(self.start_epoch, num_epochs):

            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print("-" * 60)

            # ==========================
            # Train
            # ==========================

            train_loss, train_metrics = self.train_one_epoch(
                epoch + 1
            )

            # ==========================
            # Validation
            # ==========================

            val_loss, val_metrics = self.validate(
                epoch + 1
            )

            # ==========================
            # Scheduler
            # ==========================

            if self.scheduler is not None:
                self.scheduler.step()

            # ==========================
            # Logging
            # ==========================

            print(f"Train Loss : {train_loss:.4f}")
            print(f"Val Loss   : {val_loss:.4f}")

            print(
                f"Train Acc  : {train_metrics['accuracy']:.4f}"
            )

            print(
                f"Val Acc    : {val_metrics['accuracy']:.4f}"
            )

            print(
                f"Val Macro F1 : {val_metrics['macro_f1']:.4f}"
            )

            print(
                f"Val QWK      : {val_metrics['qwk']:.4f}"
            )

            # ==========================
            # Save Best Model
            # ==========================

            current_metric = val_metrics["qwk"]

            if current_metric > self.best_metric:

                self.best_metric = current_metric

                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    best_metric=self.best_metric,
                    filename="best_model.pth"
                )

                print("Best model updated.")

            # ==========================
            # Save Last Checkpoint
            # ==========================

            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch + 1,
                best_metric=self.best_metric,
                filename="last_checkpoint.pth"
            )

        print("\nTraining Finished Successfully.")
