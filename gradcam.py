import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "outputs/matplotlib_cache")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from config import BEST_MODEL_PATH, CLASS_NAMES, IMAGE_SIZE, OUTPUT_DIR
from data_loader import get_transforms
from model import build_model
from train import get_device


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def __call__(self, image_tensor, target_class=None):
        output = self.model(image_tensor)
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        score = output[:, target_class].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class, torch.softmax(output, dim=1).detach().cpu().squeeze(0)


def denormalize(tensor):
    image = tensor.detach().cpu().squeeze(0)
    image = image * 0.5 + 0.5
    image = image.permute(1, 2, 0).numpy()
    return np.clip(image, 0, 1)


def save_gradcam(image_path, target_class=None):
    OUTPUT_DIR.mkdir(exist_ok=True)
    device = get_device()

    model = build_model().to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = get_transforms(train=False)(image).unsqueeze(0).to(device)

    gradcam = GradCAM(model, model.layer4[-1])
    cam, pred_idx, probs = gradcam(tensor, target_class=target_class)
    gradcam.remove_hooks()

    base = denormalize(tensor)
    heatmap = plt.get_cmap("jet")(cam)[..., :3]
    overlay = np.clip(0.55 * base + 0.45 * heatmap, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(base, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM: {CLASS_NAMES[pred_idx]} ({probs[pred_idx]:.3f})")
    axes[1].axis("off")

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"gradcam_{CLASS_NAMES[pred_idx]}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path, CLASS_NAMES[pred_idx], probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to a brain MRI image")
    parser.add_argument("--target-class", type=int, default=None)
    args = parser.parse_args()

    output_path, pred_class, probs = save_gradcam(args.image_path, args.target_class)
    print(f"prediction: {pred_class}")
    for class_name, prob in zip(CLASS_NAMES, probs):
        print(f"{class_name:12s}: {prob.item():.4f}")
    print(f"saved Grad-CAM: {output_path}")


if __name__ == "__main__":
    main()
