import os
import torch


class CheckpointManager:
    """
    Utility class for saving and loading training checkpoints.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save(
        self,
        model,
        optimizer,
        scheduler,
        epoch,
        best_metric,
        filename="checkpoint.pth"
    ):
        """
        Save training checkpoint.
        """

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        save_path = os.path.join(self.save_dir, filename)

        torch.save(checkpoint, save_path)

        print(f"Checkpoint saved: {save_path}")

    def load(
        self,
        model,
        optimizer=None,
        scheduler=None,
        filename="checkpoint.pth",
        map_location="cpu"
    ):
        """
        Load checkpoint.
        """

        checkpoint_path = os.path.join(self.save_dir, filename)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if optimizer is not None:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if (
            scheduler is not None and
            "scheduler_state_dict" in checkpoint
        ):
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        epoch = checkpoint["epoch"]
        best_metric = checkpoint["best_metric"]

        print(f"Checkpoint loaded: {checkpoint_path}")

        return epoch, best_metric
