import argparse

import torch
from PIL import Image

from config import BEST_MODEL_PATH, CLASS_NAMES
from data_loader import get_transforms
from model import build_model
from train import get_device


def predict(image_path):
    device = get_device()
    transform = get_transforms(train=False)

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    model = build_model().to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).squeeze(0).cpu()

    pred_idx = int(probs.argmax())
    return CLASS_NAMES[pred_idx], probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to a brain MRI image")
    args = parser.parse_args()

    pred_class, probs = predict(args.image_path)

    print(f"prediction: {pred_class}")
    for class_name, prob in zip(CLASS_NAMES, probs):
        print(f"{class_name:12s}: {prob.item():.4f}")


if __name__ == "__main__":
    main()
