"""Step 6: full fine-tuning of EfficientNet-B4 with mixed precision + augmentation.

Loads ImageNet-pretrained EfficientNet-B4, replaces the classifier head, and
fine-tunes the entire network with discriminative LRs (backbone slow, head
fast). Uses AMP for ~2x throughput, RandAugment + horizontal flip for
augmentation, label smoothing, and a cosine LR schedule. Validation and test
use horizontal-flip test-time augmentation.

First-time install (into the project venv):

    .venv/bin/python -m pip install torch torchvision

Run:

    .venv/bin/python src/6_train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, project_paths


# B4 input is 380x380. With full fine-tuning + AMP, batch 16 keeps ~70% of an
# 8 GB card. Watch nvidia-smi on the first epoch — bump up if comfortable.
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 7
LR_HEAD = 1e-3       # fresh classifier — train fast
LR_BACKBONE = 1e-5   # pretrained backbone — nudge gently so we don't lose ImageNet features
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class InsectDataset(Dataset):
    def __init__(self, csv_path: Path, root: Path, label_to_idx: dict[int, int], transform):
        df = pd.read_csv(csv_path)
        self.paths = [root / p for p in df["path"].tolist()]
        self.labels = df["taxon_id"].astype(int).map(label_to_idx).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def build_model(num_classes: int):
    weights = EfficientNet_B4_Weights.DEFAULT
    model = efficientnet_b4(weights=weights)
    # Full fine-tuning: every layer is trainable. Discriminative LRs in the
    # optimizer keep the backbone moving slowly so ImageNet features aren't
    # blown away by the large gradient from the freshly-initialized head.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_transforms():
    # Validation/test: deterministic preprocessing matching B4's pretraining
    # (resize 384 -> center-crop 380 -> ImageNet normalize).
    eval_transform = EfficientNet_B4_Weights.DEFAULT.transforms()
    # Training: RandAugment is a strong battle-tested recipe — it samples 2 of
    # 14 standard image ops at moderate strength every call. Operates on PIL,
    # so it must come before ToTensor.
    train_transform = transforms.Compose([
        transforms.Resize(384),
        transforms.RandomCrop(380),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


@torch.no_grad()
def score(model, loader, device) -> tuple[float, float]:
    """Top-1 and top-5 with horizontal-flip TTA (average softmax of orig + flipped)."""
    model.train(False)
    correct1 = correct5 = total = 0
    for x, y in tqdm(loader, desc="score", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(device_type=device, dtype=torch.float16):
            probs = model(x).softmax(dim=1) + model(torch.flip(x, dims=[-1])).softmax(dim=1)
        top5 = probs.topk(5, dim=1).indices
        correct1 += (top5[:, 0] == y).sum().item()
        correct5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.size(0)
    return correct1 / max(total, 1), correct5 / max(total, 1)


def main() -> None:
    cfg = load_config()
    paths = project_paths(cfg)
    splits = paths["splits"]
    root = paths["root"]

    label_set: set[int] = set()
    for name in ("train", "val", "test"):
        label_set.update(pd.read_csv(splits / f"{name}.csv")["taxon_id"].astype(int).tolist())
    label_to_idx = {tid: i for i, tid in enumerate(sorted(label_set))}
    num_classes = len(label_to_idx)
    print(f"[info] device={DEVICE}  classes={num_classes:,}  model=EfficientNet-B4")

    model = build_model(num_classes).to(DEVICE)
    train_transform, eval_transform = build_transforms()

    train_loader = DataLoader(
        InsectDataset(splits / "train.csv", root, label_to_idx, train_transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        InsectDataset(splits / "val.csv", root, label_to_idx, eval_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        InsectDataset(splits / "test.csv", root, label_to_idx, eval_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
    )

    # Two param groups so the cosine schedule scales each LR by the same factor
    # while preserving the 100x ratio between head and backbone.
    head_params = list(model.classifier.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]
    optim = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = GradScaler(DEVICE)

    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    with open(ckpt_dir / "label_to_idx.json", "w") as f:
        json.dump({str(k): v for k, v in label_to_idx.items()}, f)

    best_val = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train(True)
        running_loss = correct = seen = 0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{EPOCHS}", unit="batch")
        for x, y in bar:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optim.zero_grad()
            with autocast(device_type=DEVICE, dtype=torch.float16):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
            bar.set_postfix(loss=running_loss / seen, acc=correct / seen)
        scheduler.step()

        val_top1, val_top5 = score(model, val_loader, DEVICE)
        print(
            f"[epoch {epoch}] lr_bb={optim.param_groups[0]['lr']:.2e} "
            f"lr_head={optim.param_groups[1]['lr']:.2e} "
            f"train_loss={running_loss/seen:.4f} train_acc={correct/seen:.4f} "
            f"val_top1={val_top1:.4f} val_top5={val_top5:.4f}"
        )
        if val_top1 > best_val:
            best_val = val_top1
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"[ckpt] saved best.pt (val_top1={val_top1:.4f})")

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=DEVICE, weights_only=True))
    test_top1, test_top5 = score(model, test_loader, DEVICE)
    print(f"[final] test_top1={test_top1:.4f} test_top5={test_top5:.4f} (best val_top1={best_val:.4f})")


if __name__ == "__main__":
    main()
