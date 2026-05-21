"""Step 3: download photos to data/images/<taxon_id>_<slug>/<photo_id>.<ext>.

Multithreaded, resumable, with a hard disk-budget stop.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    disk_used_gb, get_session, load_config, project_paths, safe_request,
)


INAT_BASE = "https://inaturalist-open-data.s3.amazonaws.com"
BUDGET_CHECK_EVERY = 5_000


def setup_logger(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("inat.download")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(logs_dir / "download.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    return logger


def download_one(session, row, images_root: Path, size: str, timeout: int, logger):
    photo_id = int(row.photo_id)
    ext = str(row.extension)
    taxon_id = int(row.taxon_id)
    slug = str(row.slug)

    folder = images_root / f"{taxon_id}_{slug}"
    dest = folder / f"{photo_id}.{ext}"
    if dest.exists() and dest.stat().st_size > 0:
        return "skipped", photo_id

    url = f"{INAT_BASE}/photos/{photo_id}/{size}.{ext}"
    try:
        resp = safe_request(session, url, timeout=timeout, stream=True)
    except Exception as e:
        logger.error(f"{photo_id} error {type(e).__name__}: {e}")
        return "error", photo_id
    if resp is None:
        logger.info(f"{photo_id} 404")
        return "404", photo_id

    folder.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, dest)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        logger.error(f"{photo_id} write error {type(e).__name__}: {e}")
        return "error", photo_id
    finally:
        resp.close()

    return "ok", photo_id


def main() -> None:
    cfg = load_config()
    paths = project_paths(cfg)
    paths["images"].mkdir(parents=True, exist_ok=True)
    logger = setup_logger(paths["logs"])

    df = pd.read_parquet(paths["filtered"] / "photos_to_download.parquet")
    print(f"[info] candidate photos: {len(df):,}")

    # Pre-filter: drop rows whose target file already exists.
    print("[info] scanning existing files for resume...")
    def exists(row):
        return (paths["images"] / f"{int(row.taxon_id)}_{row.slug}" / f"{int(row.photo_id)}.{row.extension}").exists()
    skipped_existing = 0
    keep_idx = []
    for r in tqdm(df.itertuples(index=True), total=len(df), desc="resume scan"):
        if exists(r):
            skipped_existing += 1
        else:
            keep_idx.append(r.Index)
    remaining = df.loc[keep_idx].reset_index(drop=True)
    print(f"[info] {skipped_existing:,} already on disk, {len(remaining):,} to download")

    if remaining.empty:
        print("[done] nothing to do")
        return

    session = get_session(cfg["max_workers"])
    counters = {"ok": 0, "skipped": skipped_existing, "404": 0, "error": 0}
    lock = threading.Lock()
    stop_event = threading.Event()

    with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as pool:
        futures = {
            pool.submit(download_one, session, row, paths["images"], cfg["size"], cfg["timeout_seconds"], logger): row
            for row in remaining.itertuples(index=False)
        }
        try:
            bar = tqdm(total=len(futures), desc="downloading", unit="img")
            for fut in as_completed(futures):
                if stop_event.is_set():
                    fut.cancel()
                    continue
                try:
                    status, _ = fut.result()
                except Exception as e:
                    status = "error"
                    logger.error(f"future raised: {type(e).__name__}: {e}")
                with lock:
                    counters[status] = counters.get(status, 0) + 1
                    done_total = counters["ok"] + counters["404"] + counters["error"]
                bar.update(1)
                bar.set_postfix(ok=counters["ok"], err=counters["error"], nf=counters["404"])

                if done_total and done_total % BUDGET_CHECK_EVERY == 0:
                    used = disk_used_gb(paths["images"])
                    if used >= cfg["disk_budget_gb"]:
                        msg = (
                            f"disk budget reached: {used:.2f} GB >= "
                            f"{cfg['disk_budget_gb']} GB, stopping downloads"
                        )
                        print(f"\n[STOP] {msg}")
                        logger.warning(msg)
                        stop_event.set()
                        for f in futures:
                            f.cancel()
                        break
            bar.close()
        except KeyboardInterrupt:
            print("\n[abort] Ctrl+C — cancelling remaining work, files on disk are kept.")
            stop_event.set()
            for f in futures:
                f.cancel()

    print("\n--- Download summary ---")
    for k, v in counters.items():
        print(f"  {k:8s}: {v:,}")
    print(f"  disk used: {disk_used_gb(paths['images']):.2f} GB")
    print(f"  log:       {paths['logs'] / 'download.log'}")


if __name__ == "__main__":
    main()
