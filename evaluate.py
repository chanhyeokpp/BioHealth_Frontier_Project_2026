import os

os.environ.setdefault("MPLCONFIGDIR", "outputs/matplotlib_cache")

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from config import BEST_MODEL_PATH, CLASS_NAMES, NUM_CLASSES, OUTPUT_DIR
from data_loader import build_loaders
from model import build_model
from train import get_device


def collect_predictions(model, loader, device):
    model.eval()
    y_true = []
    y_prob = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            y_true.extend(labels.numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = y_prob.argmax(axis=1)
    return y_true, y_pred, y_prob


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_roc_curve(y_true, y_prob):
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")

    fig, ax = plt.subplots(figsize=(8, 6))
    for class_idx, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], y_prob[:, class_idx])
        class_auc = roc_auc_score(y_true_bin[:, class_idx], y_prob[:, class_idx])
        ax.plot(fpr, tpr, label=f"{class_name} AUC={class_auc:.3f}")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (macro AUC={auc:.3f})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = OUTPUT_DIR / "roc_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, auc


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    device = get_device()

    _, _, test_loader = build_loaders()
    model = build_model().to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    y_true, y_pred, y_prob = collect_predictions(model, test_loader, device)
    acc = accuracy_score(y_true, y_pred)

    print("=" * 60)
    print(f"test accuracy: {acc:.4f}")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm_path = save_confusion_matrix(y_true, y_pred)
    roc_path, auc = save_roc_curve(y_true, y_prob)

    print(f"macro ROC-AUC: {auc:.4f}")
    print(f"saved confusion matrix: {cm_path}")
    print(f"saved ROC curve: {roc_path}")


if __name__ == "__main__":
    main()
