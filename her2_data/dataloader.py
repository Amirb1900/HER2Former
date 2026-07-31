
from torch.utils.data import DataLoader

from HER2Former.datasets.dataset import HER2Dataset
from HER2Former.datasets.transforms import (
    build_train_transforms,
    build_val_transforms,
    build_test_transforms,
)


def create_dataloaders(cfg):
    """
    Create train, validation and test dataloaders.
    """

    train_dataset = HER2Dataset(
        root_dir=f"{cfg.dataset.data_dir}/{cfg.dataset.train_dir}",
        transform=build_train_transforms(cfg.dataset.image_size),
    )

    val_dataset = HER2Dataset(
        root_dir=f"{cfg.dataset.data_dir}/{cfg.dataset.val_dir}",
        transform=build_val_transforms(cfg.dataset.image_size),
    )

    test_dataset = HER2Dataset(
        root_dir=f"{cfg.dataset.data_dir}/{cfg.dataset.test_dir}",
        transform=build_test_transforms(cfg.dataset.image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )

    return train_loader, val_loader, test_loader
