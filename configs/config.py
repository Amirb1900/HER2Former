
from dataclasses import dataclass, field
from pathlib import Path
import torch


# =========================
# Dataset
# =========================

@dataclass
class DatasetConfig:
    data_dir: str = "/kaggle/input/datasets/amirb1900/resized-bci-512/512BCI_dataset"

    train_dir: str = "train/train/HE"
    val_dir: str = "val/val/HE"
    test_dir: str = "test/test/HE"

    image_size: int = 224
    num_classes: int = 4

    num_workers: int = 4
    pin_memory: bool = True


# =========================
# Model
# =========================
model_path: str = "/root/.cache/huggingface/..."
@dataclass
class ModelConfig:
    backbone: str = "Virchow2"

    embedding_dim: int = 1280

    num_patch_tokens: int = 256

    num_register_tokens: int = 4

    top_k: int = 64

    use_cross_attention: bool = True

    freeze_backbone: bool = False


# =========================
# Training
# =========================

@dataclass
class TrainingConfig:
    batch_size: int = 16

    epochs: int = 50

    learning_rate: float = 1e-4

    weight_decay: float = 1e-4

    seed: int = 42

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Logging
# =========================

@dataclass
class LoggingConfig:
    save_dir: str = "./outputs"

    checkpoint_dir: str = "./outputs/checkpoints"

    log_dir: str = "./outputs/logs"


# =========================
# Main Config
# =========================

@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    model: ModelConfig = field(default_factory=ModelConfig)

    training: TrainingConfig = field(default_factory=TrainingConfig)

    logging: LoggingConfig = field(default_factory=LoggingConfig)


cfg = Config()
