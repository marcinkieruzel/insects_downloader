# European Insect Image Dataset

A reproducible pipeline that downloads research-grade photos of European insects from
the public [iNaturalist Open Dataset](https://github.com/inaturalist/inaturalist-open-data)
on AWS S3 and produces a class-balanced, ML-ready folder structure.

Target: ~80 GB of 240 px images covering thousands of insect species across Europe,
suitable for training a 224 px image classifier.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run order

```bash
python src/1_download_metadata.py   # ~12 GB of .csv.gz + Natural Earth shapefile
python src/2_filter.py              # produces filtered/photos_to_download.parquet, deletes raw metadata
python src/3_download_photos.py     # downloads up to ~80 GB of images, multithreaded, resumable
python src/4_build_manifest.py      # verifies images, writes manifest.csv + attribution.csv + class_counts.csv
python src/5_train_test_split.py    # stratified 80/10/10 split CSVs in splits/
```

## Configuration

All knobs are in [config.yaml](config.yaml). Most important:

- `size: small` — 240 px. Switch to `medium` (500 px) only if you raise `disk_budget_gb`.
- `min_photos_per_species: 20` — drop species with fewer photos.
- `max_photos_per_species: 500` — cap large classes (random-sampled with seed).
- `disk_budget_gb: 80` — hard stop in script 3 when image folder reaches this size.
- `licenses_allowed` — defaults to all Creative Commons variants iNaturalist offers.

## Disk budget

The pipeline fits in ~92 GB peak:

- Step 1 writes ~12 GB of metadata to `inat_metadata/`.
- Step 2 deletes `inat_metadata/` after filtering (when `delete_metadata_after_filter: true`).
- Step 3 writes up to `disk_budget_gb` GB of images to `data/images/`.
- Step 3 monitors `data/images/` size every 5,000 downloads and stops gracefully if the budget is hit.

## Resume

Every script is idempotent:

- Script 1 skips metadata files that already exist and pass gzip integrity.
- Script 3 skips photos already on disk with size > 0.
- Scripts 2, 4, 5 simply overwrite their outputs.

Ctrl+C any time and re-run — no work is repeated.

## Adjusting the dataset

- **Grow it**: bump `max_photos_per_species` and rerun scripts 2 + 3. Only the new files are downloaded.
- **Shrink it**: lower the cap and manually delete the excess files — the script will not auto-prune.
- **Different geography**: edit `bbox` and `use_precise_europe_polygon`, rerun script 2 + 3.

## Output layout

```
data/
└── images/
    └── <taxon_id>_<scientific_name_slug>/
        └── <photo_id>.<ext>

manifest.csv         # one row per image with metadata
attribution.csv      # CC attribution string per image
class_counts.csv     # per-species image count
splits/{train,val,test}.csv
```

## Ethics

Most iNaturalist photos are CC-BY-NC. **Do not redistribute these images commercially.**
Always ship `attribution.csv` alongside the dataset.

## Data sources

- iNaturalist Open Dataset metadata: https://inaturalist-open-data.s3.amazonaws.com/
- Photo URLs: `https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{size}.{extension}`
- Natural Earth countries (for Europe polygon): https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip
