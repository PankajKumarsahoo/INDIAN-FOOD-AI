"""Dataset discovery, splitting, PyTorch Dataset and transforms. Everything reads from local disk."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Tolerate partially-downloaded images instead of crashing a DataLoader worker.
ImageFile.LOAD_TRUNCATED_IMAGES = True

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- discovery
def find_images(raw_dir, extensions):
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {raw_dir}\n"
            "Run `python scripts/01_download_dataset.py` or put your class folders there."
        )
    exts = {e.lower() for e in extensions}
    return sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def build_dataframe(raw_dir, extensions):
    """image_path is stored RELATIVE to raw_dir (posix style) so CSVs work on any machine/OS."""
    raw_dir = Path(raw_dir)
    rows = [
        {"image_path": p.relative_to(raw_dir).as_posix(), "class_name": p.parent.name}
        for p in find_images(raw_dir, extensions)
    ]
    return pd.DataFrame(rows)


def is_image_ok(path):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- splitting
def make_splits(df, train=0.8, val=0.1, test=0.1, seed=42):
    assert abs(train + val + test - 1.0) < 1e-6, "split fractions must sum to 1"
    train_df, temp_df = train_test_split(
        df, test_size=1 - train, random_state=seed, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=test / (val + test), random_state=seed, stratify=temp_df["label"]
    )
    return (d.reset_index(drop=True) for d in (train_df, val_df, test_df))


# ---------------------------------------------------------------- transforms
def get_transforms(image_size=224, resize_size=256):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


# ---------------------------------------------------------------- dataset
class IndianFoodDataset(Dataset):
    def __init__(self, dataframe, raw_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.raw_dir = Path(raw_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _load(self, rel_path):
        image = Image.open(self.raw_dir / rel_path).convert("RGB")
        image.load()
        return image

    def __getitem__(self, index):
        for offset in range(len(self.df)):  # skip unreadable files instead of crashing
            row = self.df.iloc[(index + offset) % len(self.df)]
            try:
                image = self._load(row["image_path"])
                break
            except (OSError, IOError):
                print(f"Skipping unreadable image: {row['image_path']}")
        else:
            raise RuntimeError("No readable images in this split")

        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])


def make_loaders(cfg, train_df, val_df, test_df):
    c = cfg["classifier"]
    raw = cfg["paths"]["raw_data"]
    train_tf, eval_tf = get_transforms(c["image_size"], c["resize_size"])
    pin = torch.cuda.is_available()
    kw = dict(batch_size=c["batch_size"], num_workers=c["num_workers"], pin_memory=pin,
              persistent_workers=c["num_workers"] > 0)
    train_loader = DataLoader(IndianFoodDataset(train_df, raw, train_tf), shuffle=True, **kw)
    val_loader = DataLoader(IndianFoodDataset(val_df, raw, eval_tf), shuffle=False, **kw)
    test_loader = DataLoader(IndianFoodDataset(test_df, raw, eval_tf), shuffle=False, **kw)
    return train_loader, val_loader, test_loader
