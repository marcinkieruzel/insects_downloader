"""Step 6 (demo): quick transfer-learning run on the top-N classes.

Same architecture as 6_train.py (EfficientNet-B3, frozen backbone, fresh head)
but filtered to the N species with the most photos and a single epoch — so the
whole loop finishes in ~10 minutes on a single GPU. Use this to learn the
mechanics before committing to the multi-day full run.

Run:

    .venv/bin/python src/6_train_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, project_paths


N_CLASSES = 50
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 1
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class InsectDataset(Dataset):
    """Dataset built from an in-memory DataFrame (so we can filter first)."""

    def __init__(self, df: pd.DataFrame, root: Path, label_to_idx: dict[int, int], transform):
        self.paths = [root / p for p in df["path"].tolist()]
        self.labels = df["taxon_id"].astype(int).map(label_to_idx).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def build_model(num_classes: int):
    weights = EfficientNet_B3_Weights.DEFAULT
    model = efficientnet_b3(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model, weights.transforms()


@torch.no_grad()
def score(model, loader, device) -> float:
    model.train(False)
    correct = total = 0
    for x, y in tqdm(loader, desc="score", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def main() -> None:
    cfg = load_config()
    paths = project_paths(cfg)
    splits = paths["splits"]
    root = paths["root"]

    # Pick the N species with the most photos and keep only those rows in every split.
    class_counts = pd.read_csv(paths["class_counts"])
    top_taxa = class_counts.nlargest(N_CLASSES, "n_photos")["taxon_id"].astype(int).tolist()
    top_set = set(top_taxa)
    label_to_idx = {tid: i for i, tid in enumerate(sorted(top_set))}
    print(f"[info] device={DEVICE}  top-{N_CLASSES} classes")

    def filtered(name: str) -> pd.DataFrame:
        df = pd.read_csv(splits / f"{name}.csv")
        df = df[df["taxon_id"].astype(int).isin(top_set)].reset_index(drop=True)
        return df

    train_df, val_df, test_df = filtered("train"), filtered("val"), filtered("test")
    print(f"[info] rows  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    model, transform = build_model(N_CLASSES)
    model.to(DEVICE)

    train_loader = DataLoader(
        InsectDataset(train_df, root, label_to_idx, transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        InsectDataset(val_df, root, label_to_idx, transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        InsectDataset(test_df, root, label_to_idx, transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
    )

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.CrossEntropyLoss()

    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    with open(ckpt_dir / "demo_label_to_idx.json", "w") as f:
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
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()
            running_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
            bar.set_postfix(loss=running_loss / seen, acc=correct / seen)

        val_acc = score(model, val_loader, DEVICE)
        print(
            f"[epoch {epoch}] train_loss={running_loss/seen:.4f} "
            f"train_acc={correct/seen:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), ckpt_dir / "demo_best.pt")
            print(f"[ckpt] saved demo_best.pt (val_acc={val_acc:.4f})")

    model.load_state_dict(torch.load(ckpt_dir / "demo_best.pt", map_location=DEVICE))
    test_acc = score(model, test_loader, DEVICE)
    print(f"[final] test_acc={test_acc:.4f} (best val_acc={best_val:.4f})")


if __name__ == "__main__":
    main()
