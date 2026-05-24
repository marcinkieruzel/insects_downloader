# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A linear 5-step batch pipeline that builds a class-balanced European insect image dataset from the public iNaturalist Open Dataset on S3. The README covers the user-facing intent (target ~80 GB of 240 px images, ML-ready folder layout); this file covers what a future Claude needs to operate the code.

## Setup and running

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/1_download_metadata.py   # ~28 GB into inat_metadata/
python src/2_filter.py              # writes filtered/*.parquet, deletes inat_metadata/ if delete_metadata_after_filter=true
python src/3_download_photos.py     # downloads up to disk_budget_gb of images into data/images/
python src/4_build_manifest.py      # writes manifest.csv, attribution.csv, class_counts.csv at project root
python src/5_train_test_split.py    # writes splits/{train,val,test}.csv
```

Scripts must run in order — each consumes the previous step's output. There are no tests, no linter, no build step. Each script is idempotent (see README §Resume): re-running steps 1 and 3 is cheap, re-running step 2 is expensive (it re-streams the full metadata).

### Long-running background runs

Steps 2 and 3 each take hours (step 2 ~2–6 h on the photos.csv.gz stream, step 3 ~many hours capped by `disk_budget_gb`). When chaining them, use the established pattern:

```bash
nohup bash -c '
  for step in 2_filter 3_download_photos 4_build_manifest 5_train_test_split; do
    num=${step%%_*}
    .venv/bin/python src/${step}.py >logs/step${num}.out 2>logs/step${num}.err || exit
  done
' >logs/chain.stdout 2>logs/chain.stderr &
```

All step logs live in [logs/](logs/). Step 3 also writes a per-photo error log at `logs/download.log`; step 4 writes `logs/manifest.log`.

## Architecture

### Single source of truth for config and paths

All knobs live in [config.yaml](config.yaml). All scripts call `utils.load_config()` (validates required keys, errors on missing) and `utils.project_paths(cfg)` (returns a dict keyed by `metadata`, `filtered`, `images`, `quarantine`, `splits`, `logs`, `manifest`, `attribution`, `class_counts`). **Always resolve paths through `project_paths()` rather than hardcoding** — `PROJECT_ROOT` is derived from `src/utils.py`'s location, so paths work regardless of the caller's CWD.

### Shared helpers in [src/utils.py](src/utils.py)

- `slugify(name)` — produces the `<slug>` half of `data/images/<taxon_id>_<slug>/`. Must stay deterministic; step 4 reconciles disk files back to metadata by `photo_id`, but the folder name is the only link between class folder and scientific name.
- `get_session(workers)` + `safe_request(...)` — every HTTP call goes through this. Retries on 5xx and connection errors with `(1, 4, 9)` second backoff; returns `None` on 404 (caller must handle). Never call `requests.get` directly.
- `disk_used_gb(path)` — step 3 calls this every `BUDGET_CHECK_EVERY=5000` downloads to decide when to stop.

### Step 2's photo-streaming OOM trap

[src/2_filter.py](src/2_filter.py) streams `photos.csv.gz` (~16 GB compressed, hundreds of 1M-row chunks) and keeps the rows whose `observation_uuid` is in the European-insect observations. The current `load_photos()` spills each matched chunk to `filtered/_photos_chunks_tmp/chunk_NNNNNN.parquet` and concatenates at the end. **Do not "simplify" this back to a `chunks.append(...)` list** — the previous version was OOM-killed mid-stream on a 30 GB host. If you change the function signature, the spill directory is passed as a parameter from `main()`.

Step 2 also has no mid-run resume — a crash means re-running from the start (the `insect_taxa.parquet` and `obs_europe_insects.parquet` files get overwritten).

### Folder layout the pipeline assumes

```
inat_metadata/             # step 1 input, deleted by step 2 if delete_metadata_after_filter=true
filtered/
  insect_taxa.parquet      # step 2 intermediate
  obs_europe_insects.parquet
  photos_to_download.parquet   # step 2 final output; step 3 and 4 input
data/
  images/<taxon_id>_<slug>/<photo_id>.<ext>   # step 3 output
  quarantine/<taxon_id>_<slug>/...            # step 4 moves unreadable images here
logs/
  step{1..5}.{out,err}     # if run with the background chain pattern
  download.log             # step 3 per-photo errors
  manifest.log             # step 4 per-file warnings
manifest.csv               # step 4 output
attribution.csv            # step 4 output
class_counts.csv           # step 4 output
splits/{train,val,test}.csv  # step 5 output
```

Steps 4 and 5 reconcile disk → metadata by parsing `<photo_id>` out of filenames and joining against `photos_to_download.parquet` — preserve that filename convention.

## Iterating on the dataset

The README's "Adjusting the dataset" section is the authoritative guide. Key invariants to know when planning changes:

- Bumping `max_photos_per_species` requires re-running step 2 (it does the sampling) and step 3 (to fetch new files). Step 3 is incremental — only new files are downloaded.
- Lowering the cap does **not** prune extra files on disk; manual cleanup needed.
- Changing `bbox` or `use_precise_europe_polygon` invalidates everything from step 2 onward.
- Step 3's `BUDGET_CHECK_EVERY` is a hard guardrail, not a soft target — the script stops the executor pool when `data/images/` size first crosses `disk_budget_gb`.
