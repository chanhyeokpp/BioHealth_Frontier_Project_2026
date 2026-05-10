import os
import nibabel as nib

DATA_DIR = "/Users/chanhyeok/Desktop/BioHealth_Frontier/BraTS2021_Training_Data"

patients = sorted([
    f for f in os.listdir(DATA_DIR)
    if f.startswith("BraTS2021")
])

print(f"총 환자 수: {len(patients)}")

shapes = set()

for p_id in patients:
    flair_path = os.path.join(
        DATA_DIR,
        p_id,
        f"{p_id}_flair.nii.gz"
    )

    img = nib.load(flair_path)
    shapes.add(img.shape)

print("발견된 shape:")
print(shapes)