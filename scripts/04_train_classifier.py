"""Phase 4: train ConvNeXt-Tiny (stage 1: head only, stage 2: full fine-tune)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from foodai.config import ensure_dirs, load_config
from foodai.data import make_loaders
from foodai.engine import train_one_epoch, validate
from foodai.model import build_classifier, save_checkpoint
from foodai.utils import get_device, load_json, seed_everything


def run_stage(name, epochs, lr, model, loaders, criterion, scaler, device, cfg, state, class_maps, params):
    train_loader, val_loader = loaders
    c = cfg["classifier"]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=c["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    for epoch in range(epochs):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, c["amp"])
        vl, va = validate(model, val_loader, criterion, device, c["amp"])
        scheduler.step()

        state["history"].append({"stage": name, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va})
        print(f"[{name}] epoch {epoch+1}/{epochs} | train loss {tl:.4f} acc {ta:.4f} "
              f"| val loss {vl:.4f} acc {va:.4f} | {time.time()-t0:.0f}s")

        if va > state["best_val_acc"]:
            state["best_val_acc"] = va
            save_checkpoint(cfg["paths"]["checkpoints"] / "best.pth", model, *class_maps, val_accuracy=va)
            print("  -> best model saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs-head", type=int, default=None)
    parser.add_argument("--epochs-finetune", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    c = cfg["classifier"]
    if args.epochs_head is not None:
        c["head"]["epochs"] = args.epochs_head
    if args.epochs_finetune is not None:
        c["finetune"]["epochs"] = args.epochs_finetune
    if args.batch_size is not None:
        c["batch_size"] = args.batch_size

    seed_everything(cfg["split"]["seed"])
    device = get_device()
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "(no GPU found - training will be slow)")

    proc = cfg["paths"]["processed"]
    try:
        train_df, val_df, test_df = (pd.read_csv(proc / f"{n}.csv") for n in ("train", "val", "test"))
    except FileNotFoundError:
        raise SystemExit("Split CSVs not found. Run: python scripts/03_prepare_data.py")

    mapping = load_json(proc / "class_mapping.json")
    class_to_idx = mapping["class_to_idx"]
    idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}

    train_loader, val_loader, _ = make_loaders(cfg, train_df, val_df, test_df)
    model = build_classifier(len(class_to_idx), pretrained=c["pretrained"]).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=c["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda", enabled=c["amp"] and device.type == "cuda")
    state = {"history": [], "best_val_acc": 0.0}
    maps = (class_to_idx, idx_to_class)

    # Stage 1: head only
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    run_stage("head", c["head"]["epochs"], c["head"]["lr"], model, (train_loader, val_loader),
              criterion, scaler, device, cfg, state, maps, [p for p in model.parameters() if p.requires_grad])

    # Stage 2: full fine-tune, starting from the best head checkpoint
    best = cfg["paths"]["checkpoints"] / "best.pth"
    if best.exists():
        model.load_state_dict(torch.load(best, map_location=device)["model_state_dict"])
    for p in model.parameters():
        p.requires_grad = True
    run_stage("finetune", c["finetune"]["epochs"], c["finetune"]["lr"], model, (train_loader, val_loader),
              criterion, scaler, device, cfg, state, maps, list(model.parameters()))

    # Save final = best checkpoint, copied to models/
    final_path = cfg["paths"]["models"] / "food_classifier_convnext_tiny.pth"
    ckpt = torch.load(best, map_location="cpu")
    torch.save(ckpt, final_path)
    print(f"Final classifier (val acc {ckpt['val_accuracy']:.4f}) saved to {final_path}")

    # history + curves
    hist = pd.DataFrame(state["history"])
    hist.to_csv(cfg["paths"]["reports"] / "training_history.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(hist["train_loss"], label="Train"); ax[0].plot(hist["val_loss"], label="Val"); ax[0].set_title("Loss")
    ax[1].plot(hist["train_acc"], label="Train"); ax[1].plot(hist["val_acc"], label="Val"); ax[1].set_title("Accuracy")
    for a in ax:
        a.set_xlabel("Epoch"); a.legend()
    plt.tight_layout()
    plt.savefig(cfg["paths"]["plots"] / "training_curves.png", dpi=120)
    print("Next: python scripts/05_evaluate.py")


if __name__ == "__main__":
    main()
