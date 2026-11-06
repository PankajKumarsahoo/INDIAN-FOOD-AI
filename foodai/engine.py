"""Training / validation loops."""
import torch
from tqdm import tqdm


def _autocast(device, enabled):
    return torch.amp.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp=True):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, amp=True):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="val", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with _autocast(device, amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        loss_sum += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    preds, labels_all = [], []
    for images, labels in tqdm(loader, desc="test", leave=False):
        out = model(images.to(device))
        preds.extend(out.argmax(1).cpu().tolist())
        labels_all.extend(labels.tolist())
    return labels_all, preds
