import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_SIZE,
    NUM_WORKERS,
    RANDOM_SEED,
    TEST_DIR,
    TRAIN_DIR,
    VAL_RATIO,
)


def get_transforms(train=True):
    if train:
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_datasets():
    full_train_ds = datasets.ImageFolder(TRAIN_DIR, transform=get_transforms(train=True))
    test_ds = datasets.ImageFolder(TEST_DIR, transform=get_transforms(train=False))

    if full_train_ds.classes != CLASS_NAMES:
        raise ValueError(
            f"Unexpected classes: {full_train_ds.classes}. Expected: {CLASS_NAMES}"
        )

    val_size = int(len(full_train_ds) * VAL_RATIO)
    train_size = len(full_train_ds) - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_ds, val_ds = random_split(
        full_train_ds,
        [train_size, val_size],
        generator=generator,
    )

    # Validation must not use random augmentation.
    val_ds.dataset = datasets.ImageFolder(TRAIN_DIR, transform=get_transforms(train=False))

    return train_ds, val_ds, test_ds


def build_loaders():
    train_ds, val_ds, test_ds = build_datasets()

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_loaders()
    print(f"classes: {CLASS_NAMES}")
    print(f"train batches: {len(train_loader)}")
    print(f"val batches: {len(val_loader)}")
    print(f"test batches: {len(test_loader)}")
