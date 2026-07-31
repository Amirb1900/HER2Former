
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset


class HER2Dataset(Dataset):
    """
    HER2 Dataset

    Returns
    -------
    image
    label
    filename
    """

    def __init__(self, root_dir, transform=None):

        self.root_dir = Path(root_dir)
        self.transform = transform

        self.image_paths = sorted(
            self.root_dir.rglob("*.png")
        )

    def __len__(self):

        return len(self.image_paths)

    @staticmethod
    def extract_label(filename):

        filename = Path(filename).stem

        label = filename.split("_")[-1]

        label = label.replace("+", "")

        return int(label)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("RGB")

        label = self.extract_label(image_path.name)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, image_path.name
