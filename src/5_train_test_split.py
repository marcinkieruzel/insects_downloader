"""Step 5: stratified train/val/test split over manifest.csv, by taxon_id."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, project_paths


def main() -> None:
    cfg = load_config()
    paths = project_paths(cfg)
    paths["splits"].mkdir(parents=True, exist_ok=True)

    ratios = cfg.get("train_val_test_ratios", [0.8, 0.1, 0.1])
    if abs(sum(ratios) - 1.0) > 1e-6 or len(ratios) != 3:
        raise ValueError(f"train_val_test_ratios must be 3 fractions summing to 1, got {ratios}")
    train_r, val_r, test_r = ratios

    manifest = pd.read_csv(paths["manifest"])
    print(f"[info] manifest rows: {len(manifest):,}, species: {manifest['taxon_id'].nunique():,}")

    counts = manifest.groupby("taxon_id").size()
    drop_taxa = counts[counts < 3].index
    if len(drop_taxa):
        print(f"[info] dropping {len(drop_taxa):,} species with <3 images (cannot stratify)")
        manifest = manifest[~manifest["taxon_id"].isin(drop_taxa)]

    seed = cfg["random_seed"]
    train, temp = train_test_split(
        manifest, test_size=(val_r + test_r),
        stratify=manifest["taxon_id"], random_state=seed,
    )
    val_share_of_temp = val_r / (val_r + test_r)
    val, test = train_test_split(
        temp, train_size=val_share_of_temp,
        stratify=temp["taxon_id"], random_state=seed,
    )

    cols = ["path", "taxon_id", "scientific_name"]
    for name, df in [("train", train), ("val", val), ("test", test)]:
        out = paths["splits"] / f"{name}.csv"
        df[cols].to_csv(out, index=False)
        print(f"[save] {out} ({len(df):,} rows, {df['taxon_id'].nunique():,} species)")

    total = len(train) + len(val) + len(test)
    print(f"\n--- Split summary ---")
    print(f"  train: {len(train):,} ({len(train)/total:.1%})")
    print(f"  val:   {len(val):,} ({len(val)/total:.1%})")
    print(f"  test:  {len(test):,} ({len(test)/total:.1%})")


if __name__ == "__main__":
    main()
