import os
import nibabel as nib
import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# ============================================================
# PATH
# ============================================================

DATA_DIR = "/Users/chanhyeok/Desktop/BioHealth_Frontier/BraTS2021_Training_Data"

# ============================================================
# SPLIT
# ============================================================

def get_split_lists():

    patients = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.startswith("BraTS2021")
    ])

    train_ids, temp_ids = train_test_split(
        patients,
        test_size=0.3,
        random_state=42
    )

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.5,
        random_state=42
    )

    return train_ids, val_ids, test_ids

# ============================================================
# DATASET
# ============================================================

class BraTS2DDataset(Dataset):

    def __init__(self, patient_ids, data_dir):

        self.data_dir = data_dir
        self.patient_ids = patient_ids

        self.modalities = ["flair", "t1", "t1ce", "t2"]

        self.samples = []

        print("📦 slice index 생성 중...")

        for pid in patient_ids:

            seg_path = os.path.join(
                data_dir, pid, f"{pid}_seg.nii.gz"
            )

            seg = nib.load(seg_path).get_fdata()

            for z in range(seg.shape[2]):

                # tumor 있는 slice만
                if np.sum(seg[:, :, z]) > 0:
                    self.samples.append((pid, z))

        print(f"✅ 총 slice 수: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def normalize(self, img):

        mask = img > 0

        if np.any(mask):
            img[mask] = (
                img[mask] - img[mask].mean()
            ) / (img[mask].std() + 1e-8)

        return img

    def __getitem__(self, idx):

        pid, z = self.samples[idx]
        path = os.path.join(self.data_dir, pid)

        channels = []

        for m in self.modalities:

            img_path = os.path.join(
                path, f"{pid}_{m}.nii.gz"
            )

            img = nib.load(img_path).get_fdata()

            slice_2d = img[:, :, z]
            slice_2d = self.normalize(slice_2d)

            channels.append(slice_2d)

        image = np.stack(channels, axis=0)

        seg_path = os.path.join(
            path, f"{pid}_seg.nii.gz"
        )

        mask = nib.load(seg_path).get_fdata()[:, :, z]

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.long)
        )