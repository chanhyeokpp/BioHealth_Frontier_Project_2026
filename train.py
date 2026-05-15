import random

import numpy as np
import torch
import torch.nn as nn

from config import (
    BEST_MODEL_PATH,
    EPOCHS,
    LEARNING_RATE,
    MODEL_DIR,
    RANDOM_SEED,
    WEIGHT_DECAY,
)
from data_loader import build_loaders
from model import build_model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_one_epoch(model, loader, criterion, device, optimizer=None, phase="train"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(dim=1)
            loss_sum += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == len(loader):
                current_loss = loss_sum / total
                current_acc = correct / total
                print(
                    f"{phase} batch {batch_idx:03d}/{len(loader)} | "
                    f"loss {current_loss:.4f} acc {current_acc:.4f}"
                )

    return loss_sum / total, correct / total


def main():
    set_seed(RANDOM_SEED)
    MODEL_DIR.mkdir(exist_ok=True)

    device = get_device()
    print("=" * 60)
    print(f"device: {device}")
    print("=" * 60)

    train_loader, val_loader, _ = build_loaders()
    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_one_epoch(
            model, train_loader, criterion, device, optimizer, phase="train"
        )
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, device, phase="val")

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"saved best model: {BEST_MODEL_PATH} ({best_val_acc:.4f})")

    print("=" * 60)
    print(f"best val accuracy: {best_val_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
