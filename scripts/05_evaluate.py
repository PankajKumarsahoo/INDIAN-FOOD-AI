"""Phase 4b: evaluate the saved classifier on the held-out test split."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from foodai.config import ensure_dirs, load_config
from foodai.data import make_loaders
from foodai.engine import predict_all
from foodai.model import load_classifier
from foodai.utils import get_device


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    device = get_device()
    model, idx_to_class, _ = load_classifier(cfg["paths"]["models"] / "food_classifier_convnext_tiny.pth", device)

    proc = cfg["paths"]["processed"]
    train_df, val_df, test_df = (pd.read_csv(proc / f"{n}.csv") for n in ("train", "val", "test"))
    _, _, test_loader = make_loaders(cfg, train_df, val_df, test_df)

    y_true, y_pred = predict_all(model, test_loader, device)
    acc = accuracy_score(y_true, y_pred)
    names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"Test accuracy: {acc:.4f}")

    report = classification_report(y_true, y_pred, labels=list(range(len(names))),
                                   target_names=names, zero_division=0, output_dict=True)
    pd.DataFrame(report).T.to_csv(cfg["paths"]["reports"] / "classification_report.csv")
    print(classification_report(y_true, y_pred, labels=list(range(len(names))), target_names=names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    pd.DataFrame(cm, index=names, columns=names).to_csv(cfg["paths"]["reports"] / "confusion_matrix.csv")
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    size = max(8, len(names) * 0.25)
    plt.figure(figsize=(size, size))
    plt.imshow(cmn, cmap="Blues")
    plt.title(f"Normalized confusion matrix (test acc {acc:.3f})")
    plt.xlabel("Predicted"); plt.ylabel("True")
    if len(names) <= 60:
        plt.xticks(range(len(names)), names, rotation=90, fontsize=6)
        plt.yticks(range(len(names)), names, fontsize=6)
    plt.tight_layout()
    plt.savefig(cfg["paths"]["plots"] / "confusion_matrix.png", dpi=120)
    print("Reports saved to", cfg["paths"]["reports"])


if __name__ == "__main__":
    main()
