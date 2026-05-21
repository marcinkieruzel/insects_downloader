"""Shared helpers for the iNaturalist insect pipeline."""

from __future__ import annotations

import gzip
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional

import requests
import yaml
from requests.adapters import HTTPAdapter


PROJECT_ROOT = Path(__file__).resolve().parent.parent


_slug_re = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    if not name:
        return "unknown"
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return _slug_re.sub("_", normalized.lower()).strip("_") or "unknown"


def load_config(path: str | Path = "config.yaml") -> dict:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    required = {
        "size", "max_workers", "timeout_seconds", "licenses_allowed",
        "min_photos_per_species", "max_photos_per_species", "random_seed",
        "bbox", "output_root", "metadata_dir", "filtered_dir", "disk_budget_gb",
    }
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"config.yaml missing keys: {sorted(missing)}")
    if cfg["size"] not in {"original", "large", "medium", "small", "thumb", "square"}:
        raise ValueError(f"invalid size: {cfg['size']}")
    return cfg


def project_paths(cfg: dict) -> dict[str, Path]:
    return {
        "root": PROJECT_ROOT,
        "metadata": PROJECT_ROOT / cfg["metadata_dir"],
        "filtered": PROJECT_ROOT / cfg["filtered_dir"],
        "images": PROJECT_ROOT / cfg["output_root"],
        "quarantine": PROJECT_ROOT / "data" / "quarantine",
        "splits": PROJECT_ROOT / "splits",
        "logs": PROJECT_ROOT / "logs",
        "manifest": PROJECT_ROOT / "manifest.csv",
        "attribution": PROJECT_ROOT / "attribution.csv",
        "class_counts": PROJECT_ROOT / "class_counts.csv",
    }


def get_session(workers: int) -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=workers, pool_maxsize=workers, max_retries=0)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "inat-european-insects-pipeline/1.0"})
    return session


class HTTPNotFound(Exception):
    pass


def safe_request(
    session: requests.Session,
    url: str,
    timeout: int,
    retries: int = 3,
    backoff: tuple[int, ...] = (1, 4, 9),
    stream: bool = False,
) -> Optional[requests.Response]:
    """GET with retries on connection errors and 5xx. Returns None on 404."""
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=timeout, stream=stream)
            if resp.status_code == 404:
                resp.close()
                return None
            if 500 <= resp.status_code < 600:
                resp.close()
                last_exc = RuntimeError(f"HTTP {resp.status_code}")
            else:
                resp.raise_for_status()
                return resp
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            last_exc = e
        if attempt < retries - 1:
            time.sleep(backoff[min(attempt, len(backoff) - 1)])
    if last_exc:
        raise last_exc
    return None


def disk_used_gb(path: str | Path) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total / 1e9


def verify_gzip(path: str | Path) -> bool:
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 20):
                pass
        return True
    except (OSError, EOFError):
        return False


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.2f} {u}"
        f /= 1024
    return f"{f:.2f} TB"
