import torch
import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


def build_classifier(num_classes, pretrained=True):
    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def save_checkpoint(path, model, class_to_idx, idx_to_class, **extra):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
            "architecture": "convnext_tiny",
            **extra,
        },
        path,
    )


def load_classifier(path, device):
    ckpt = torch.load(path, map_location=device)
    idx_to_class = {int(k): v for k, v in ckpt["idx_to_class"].items()}
    model = build_classifier(len(idx_to_class), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), idx_to_class, ckpt
