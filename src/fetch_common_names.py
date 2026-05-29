"""Fetch Polish vernacular names for the dataset's species from the iNaturalist API.

Reads the taxon ids from class_counts.csv (the model's class set) and writes
common_names.csv (taxon_id,common_name) at the project root. The Flask app
(www/app.py) joins this onto its predictions; species without a Polish name fall
back to the scientific name, so this file is optional at serve time.

One-shot and idempotent — re-running overwrites common_names.csv.

    .venv/bin/python src/fetch_common_names.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_session, load_config, project_paths, safe_request

API_URL = "https://api.inaturalist.org/v1/taxa/{ids}"
LOCALE = "pl"
BATCH_SIZE = 30          # iNaturalist accepts up to 30 ids per /taxa request
SLEEP_BETWEEN = 1.0      # iNat asks for ~1 req/s sustained, < 10k/day


def main() -> None:
    cfg = load_config()
    paths = project_paths(cfg)
    timeout = cfg["timeout_seconds"]

    counts = pd.read_csv(paths["class_counts"])
    taxon_ids = sorted(int(t) for t in counts["taxon_id"].unique())
    print(f"[info] {len(taxon_ids):,} taxa from {paths['class_counts'].name}")

    session = get_session(workers=1)
    batches = [taxon_ids[i:i + BATCH_SIZE] for i in range(0, len(taxon_ids), BATCH_SIZE)]
    names: dict[int, str] = {}

    for n, batch in enumerate(batches, start=1):
        url = API_URL.format(ids=",".join(str(t) for t in batch)) + f"?locale={LOCALE}"
        resp = safe_request(session, url, timeout=timeout)
        if resp is None:
            print(f"[warn] batch {n}/{len(batches)} returned 404, skipping")
            continue
        for r in resp.json().get("results", []):
            common = r.get("preferred_common_name")
            if common:
                names[int(r["id"])] = common.strip()
        resp.close()

        if n % 10 == 0 or n == len(batches):
            print(f"[info] batch {n}/{len(batches)} — {len(names):,} names so far")
        if n < len(batches):
            time.sleep(SLEEP_BETWEEN)

    out = pd.DataFrame(
        sorted(names.items()), columns=["taxon_id", "common_name"]
    )
    out.to_csv(paths["common_names"], index=False)
    pct = 100 * len(names) / len(taxon_ids) if taxon_ids else 0
    print(
        f"[done] {len(names):,}/{len(taxon_ids):,} ({pct:.1f}%) species have a "
        f"Polish name -> {paths['common_names']}"
    )


if __name__ == "__main__":
    main()
