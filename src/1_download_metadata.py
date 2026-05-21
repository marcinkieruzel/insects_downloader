"""Step 1: download iNaturalist Open Dataset metadata + Natural Earth countries shapefile."""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    get_session, human_bytes, load_config, project_paths,
    safe_request, verify_gzip,
)


INAT_BASE = "https://inaturalist-open-data.s3.amazonaws.com"
METADATA_FILES = ["taxa.csv.gz", "observations.csv.gz", "photos.csv.gz", "observers.csv.gz"]

NE_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
NE_DIRNAME = "ne_110m_admin_0_countries"


def download_stream(session, url: str, dest: Path, timeout: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    resp = safe_request(session, url, timeout=timeout, stream=True)
    if resp is None:
        raise RuntimeError(f"404 at {url}")
    total = int(resp.headers.get("Content-Length", 0)) or None
    # S3 serves *.csv.gz with Content-Encoding: gzip, so iter_content would
    # transparently decompress and write plain CSV under the .gz name. Read
    # from resp.raw with decode_content=False to preserve the gzip bytes.
    with open(tmp, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=dest.name
    ) as bar:
        for chunk in iter(lambda: resp.raw.read(1 << 20, decode_content=False), b""):
            f.write(chunk)
            bar.update(len(chunk))
    resp.close()
    os.replace(tmp, dest)


def ensure_metadata_file(session, name: str, metadata_dir: Path, timeout: int) -> None:
    dest = metadata_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {name} already present ({human_bytes(dest.stat().st_size)}), verifying gzip...")
        if verify_gzip(dest):
            print(f"[ok]   {name} gzip integrity OK")
            return
        print(f"[warn] {name} failed gzip verify, redownloading")
        dest.unlink()

    url = f"{INAT_BASE}/{name}"
    print(f"[get]  {url}")
    download_stream(session, url, dest, timeout=timeout)
    if not verify_gzip(dest):
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"gzip verify failed for {name}")
    print(f"[ok]   {name} verified ({human_bytes(dest.stat().st_size)})")


def ensure_natural_earth(session, metadata_dir: Path, timeout: int) -> None:
    out_dir = metadata_dir / NE_DIRNAME
    shp = out_dir / f"{NE_DIRNAME}.shp"
    if shp.exists() and shp.stat().st_size > 0:
        print(f"[skip] Natural Earth shapefile already present at {shp}")
        return

    zip_path = metadata_dir / "ne_110m_admin_0_countries.zip"
    print(f"[get]  {NE_URL}")
    download_stream(session, NE_URL, zip_path, timeout=timeout)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    zip_path.unlink()
    if not shp.exists():
        raise RuntimeError(f"expected {shp} after extraction")
    print(f"[ok]   Natural Earth extracted to {out_dir}")


def main() -> None:
    cfg = load_config()
    paths = project_paths(cfg)
    paths["metadata"].mkdir(parents=True, exist_ok=True)

    session = get_session(workers=4)
    for name in METADATA_FILES:
        ensure_metadata_file(session, name, paths["metadata"], cfg["timeout_seconds"])
    ensure_natural_earth(session, paths["metadata"], cfg["timeout_seconds"])

    print("\n--- Metadata summary ---")
    total = 0
    for p in sorted(paths["metadata"].rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            total += size
            print(f"  {p.relative_to(paths['metadata'])}: {human_bytes(size)}")
    print(f"  TOTAL: {human_bytes(total)}")


if __name__ == "__main__":
    main()
