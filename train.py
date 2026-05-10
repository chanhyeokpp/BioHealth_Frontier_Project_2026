import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import (
    get_split_lists,
    BraTS2DDataset,
    DATA_DIR
)

from model import UNet2D

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("=" * 60)
print(f"🚀 device: {device}")
print("=" * 60)

# ============================================================
# DICE
# ============================================================

def dice_score(pred, target, num_classes=5):

    dice = 0.0

    pred = pred.view(-1)
    target = target.view(-1)

    for c in range(1, num_classes):

        p = (pred == c)
        t = (target == c)

        inter = (p & t).sum().float()
        union = p.sum() + t.sum()

        if union == 0:
            continue

        dice += (2 * inter) / union

    return dice / (num_classes - 1)

# ============================================================
# VALIDATION
# ============================================================

def validate(model, loader, criterion):

    model.eval()
    loss_sum = 0

    with torch.no_grad():

        for x, y in loader:

            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)

            loss_sum += loss.item()

    return loss_sum / len(loader)

# ============================================================
# DATA
# ============================================================

train_ids, val_ids, test_ids = get_split_lists()

train_ids = train_ids[:20]
val_ids = val_ids[:5]
test_ids = test_ids[:5]

train_ds = BraTS2DDataset(train_ids, DATA_DIR)
val_ds = BraTS2DDataset(val_ids, DATA_DIR)
test_ds = BraTS2DDataset(test_ids, DATA_DIR)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=2)
test_loader = DataLoader(test_ds, batch_size=1)

# ============================================================
# MODEL
# ============================================================

model = UNet2D().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ============================================================
# TRAIN
# ============================================================

EPOCHS = 1

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for i, (x, y) in enumerate(train_loader):

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    val_loss = validate(model, val_loader, criterion)

    print("=" * 60)
    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Val Loss:   {val_loss:.4f}")

    torch.save(model.state_dict(), f"unet_epoch_{epoch+1}.pth")

# ============================================================
# TEST (DICE)
# ============================================================

model.eval()

dice_list = []

with torch.no_grad():

    for x, y in test_loader:

        x, y = x.to(device), y.to(device)

        out = model(x)
        pred = torch.argmax(out, dim=1)

        dice = dice_score(pred, y)
        dice_list.append(dice.item())

print("=" * 60)
print(f"🔥 TEST DICE: {sum(dice_list)/len(dice_list):.4f}")
print("=" * 60)