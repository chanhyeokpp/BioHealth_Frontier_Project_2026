# inference.py

import os

import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

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
# MODEL LOAD
# ============================================================

model = UNet2D().to(device)

model.load_state_dict(
    torch.load(
        "unet2d_epoch_1.pth",
        map_location=device
    )
)

model.eval()

print("✅ 모델 로드 완료")

# ============================================================
# DATA PATH
# ============================================================

DATA_DIR = "/Users/chanhyeok/Desktop/BioHealth_Frontier/BraTS2021_Training_Data"

patient_id = "BraTS2021_00000"

patient_path = os.path.join(
    DATA_DIR,
    patient_id
)

# ============================================================
# MRI LOAD
# ============================================================

modalities = [
    "flair",
    "t1",
    "t1ce",
    "t2"
]

# z축 몇 번째 slice 볼지
z = 80

channels = []

for m in modalities:

    img_path = os.path.join(
        patient_path,
        f"{patient_id}_{m}.nii.gz"
    )

    img = nib.load(img_path).get_fdata()

    # --------------------------------------------------------
    # 2D slice 추출
    # --------------------------------------------------------

    slice_2d = img[:, :, z]

    # --------------------------------------------------------
    # normalization
    # --------------------------------------------------------

    mask = slice_2d > 0

    if np.any(mask):

        slice_2d[mask] = (
            slice_2d[mask] - slice_2d[mask].mean()
        ) / (
            slice_2d[mask].std() + 1e-8
        )

    channels.append(slice_2d)

# ============================================================
# INPUT TENSOR
# ============================================================

image = np.stack(
    channels,
    axis=0
)

# shape:
# [4, 240, 240]

image_tensor = torch.tensor(
    image,
    dtype=torch.float32
).unsqueeze(0).to(device)

# shape:
# [1, 4, 240, 240]

print(f"✅ 입력 tensor shape: {image_tensor.shape}")

# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    output = model(image_tensor)

# output:
# [1, 5, 240, 240]

pred_mask = torch.argmax(
    output,
    dim=1
)

# shape:
# [1, 240, 240]

pred_mask = pred_mask.squeeze().cpu().numpy()

print(f"✅ 예측 mask shape: {pred_mask.shape}")

# ============================================================
# GROUND TRUTH LOAD
# ============================================================

seg_path = os.path.join(
    patient_path,
    f"{patient_id}_seg.nii.gz"
)

gt = nib.load(seg_path).get_fdata()

gt_slice = gt[:, :, z]

# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)

# ------------------------------------------------------------
# MRI
# ------------------------------------------------------------

axes[0].imshow(
    channels[0],
    cmap="gray"
)

axes[0].set_title("MRI (FLAIR)")

# ------------------------------------------------------------
# Ground Truth
# ------------------------------------------------------------

axes[1].imshow(
    gt_slice,
    cmap="jet"
)

axes[1].set_title("Ground Truth")

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

axes[2].imshow(
    pred_mask,
    cmap="jet"
)

axes[2].set_title("Prediction")

plt.tight_layout()

plt.show()