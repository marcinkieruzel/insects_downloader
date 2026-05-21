"""Step 4: walk data/images, verify with PIL, write manifest + attribution + class counts."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import disk_used_gb, load_config, project_paths


def verify_and_size(p: Path) -> tuple[int, int] | None:
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            return im.size  # (width, height)
    except Exception:
        return None


def main() -> None:
    cfg = load_config()
    paths = project_paths(cfg)
    paths["quarantine"].mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("inat.manifest")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    paths["logs"].mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(paths["logs"] / "manifest.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    meta = pd.read_parquet(paths["filtered"] / "photos_to_download.parquet")
    meta["photo_id"] = meta["photo_id"].astype("int64")
    meta_by_pid = meta.set_index("photo_id")

    files = [p for p in paths["images"].rglob("*") if p.is_file()]
    print(f"[info] {len(files):,} files under {paths['images']}")

    rows = []
    quarantined = 0
    for p in tqdm(files, desc="verify", unit="img"):
        size = verify_and_size(p)
        if size is None:
            qdest = paths["quarantine"] / p.relative_to(paths["images"])
            qdest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(qdest))
            logger.warning(f"quarantined: {p}")
            quarantined += 1
            continue
        try:
            pid = int(p.stem)
        except ValueError:
            logger.warning(f"unparseable photo_id from filename: {p}")
            continue
        if pid not in meta_by_pid.index:
            logger.warning(f"photo_id not in metadata: {p}")
            continue
        m = meta_by_pid.loc[pid]
        # handle the rare case of duplicate photo_ids (defensive)
        if isinstance(m, pd.DataFrame):
            m = m.iloc[0]
        rows.append({
            "path": str(p.relative_to(paths["root"])),
            "photo_id": pid,
            "taxon_id": int(m["taxon_id"]),
            "scientific_name": m["scientific_name"],
            "license": m["license"],
            "observer_id": int(m["observer_id"]) if pd.notna(m["observer_id"]) else None,
            "latitude": m["latitude"],
            "longitude": m["longitude"],
            "observed_on": m["observed_on"],
            "width": size[0],
            "height": size[1],
        })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(paths["manifest"], index=False)
    print(f"[save] {paths['manifest']} ({len(manifest):,} rows)")

    attribution = manifest[["photo_id", "license", "observer_id"]].copy()
    attribution["attribution_string"] = attribution.apply(
        lambda r: f"observer #{int(r.observer_id) if pd.notna(r.observer_id) else 'unknown'} "
                  f"(photo {int(r.photo_id)}), iNaturalist, {r.license}",
        axis=1,
    )
    attribution.to_csv(paths["attribution"], index=False)
    print(f"[save] {paths['attribution']}")

    counts = (
        manifest.groupby(["taxon_id", "scientific_name"]).size()
        .rename("n_photos").reset_index().sort_values("n_photos", ascending=False)
    )
    counts.to_csv(paths["class_counts"], index=False)
    print(f"[save] {paths['class_counts']}")

    print("\n--- Manifest summary ---")
    print(f"  images:      {len(manifest):,}")
    print(f"  species:     {manifest['taxon_id'].nunique():,}")
    print(f"  quarantined: {quarantined:,}")
    print(f"  disk used:   {disk_used_gb(paths['images']):.2f} GB")


if __name__ == "__main__":
    main()
