
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transforms(image_size: int = 224):

    return transforms.Compose([

        transforms.Resize((image_size, image_size)),

        transforms.RandomHorizontalFlip(),

        transforms.RandomVerticalFlip(),

        transforms.RandomRotation(90),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),

    ])


def build_val_transforms(image_size: int = 224):

    return transforms.Compose([

        transforms.Resize((image_size, image_size)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),

    ])


def build_test_transforms(image_size: int = 224):

    return build_val_transforms(image_size)
