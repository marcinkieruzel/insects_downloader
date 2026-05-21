"""Step 2: filter iNaturalist metadata to European insect photos and balance per species."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config, project_paths, slugify


INSECTA_TAXON_ID = 47158

# Average bytes per image at each iNaturalist size (rough field estimates).
SIZE_BYTES_ESTIMATE = {
    "thumb": 8_000,
    "square": 6_000,
    "small": 25_000,
    "medium": 100_000,
    "large": 250_000,
    "original": 600_000,
}


def _read_taxa(taxa_path: Path) -> pd.DataFrame:
    # The iNat Open Data schema has used both 'id' and 'taxon_id' for taxa over time.
    # Probe the header to pick the right column name.
    header = pd.read_csv(taxa_path, sep="\t", nrows=0)
    id_col = "taxon_id" if "taxon_id" in header.columns else "id"
    usecols = [id_col, "ancestry", "rank_level", "rank", "name", "active"]
    usecols = [c for c in usecols if c in header.columns]
    taxa = pd.read_csv(
        taxa_path, sep="\t", usecols=usecols,
        dtype={id_col: "Int64", "ancestry": "string", "rank": "string", "name": "string"},
        low_memory=False,
    )
    if id_col != "taxon_id":
        taxa = taxa.rename(columns={id_col: "taxon_id"})
    return taxa


def build_insect_taxa(metadata_dir: Path, filtered_dir: Path) -> pd.DataFrame:
    taxa = _read_taxa(metadata_dir / "taxa.csv.gz")
    print(f"[info] total taxa rows: {len(taxa):,}")

    mask = (
        taxa["ancestry"].fillna("").str.contains(rf"(?:^|/){INSECTA_TAXON_ID}(?:/|$)", regex=True)
        | (taxa["taxon_id"] == INSECTA_TAXON_ID)
    )
    insects = taxa.loc[mask, ["taxon_id", "name", "rank"]].copy()
    insects = insects.dropna(subset=["taxon_id", "name"])
    print(f"[info] insect taxa: {len(insects):,}")

    out = filtered_dir / "insect_taxa.parquet"
    insects.to_parquet(out, index=False)
    print(f"[save] {out}")
    return insects


def filter_observations(
    metadata_dir: Path, insect_ids: set[int], bbox: tuple[float, float, float, float],
) -> pd.DataFrame:
    lat_min, lon_min, lat_max, lon_max = bbox
    obs_path = metadata_dir / "observations.csv.gz"
    print(f"[load] {obs_path} (streaming chunks of 1,000,000)")

    keep_cols = [
        "observation_uuid", "observer_id", "latitude", "longitude",
        "positional_accuracy", "taxon_id", "quality_grade", "observed_on",
    ]
    chunks: list[pd.DataFrame] = []
    total_rows = 0
    surviving = 0

    reader = pd.read_csv(
        obs_path, sep="\t", chunksize=1_000_000, usecols=keep_cols,
        dtype={
            "observation_uuid": "string", "observer_id": "Int64",
            "latitude": "float64", "longitude": "float64",
            "positional_accuracy": "Int64", "taxon_id": "Int64",
            "quality_grade": "string", "observed_on": "string",
        },
        low_memory=False,
    )
    for chunk in tqdm(reader, desc="observations", unit=" chunk"):
        total_rows += len(chunk)
        chunk = chunk.dropna(subset=["latitude", "longitude", "taxon_id"])
        chunk = chunk[chunk["quality_grade"] == "research"]
        chunk = chunk[chunk["taxon_id"].isin(insect_ids)]
        chunk = chunk[
            (chunk["latitude"].between(lat_min, lat_max))
            & (chunk["longitude"].between(lon_min, lon_max))
        ]
        if not chunk.empty:
            chunks.append(chunk)
            surviving += len(chunk)

    print(f"[info] scanned {total_rows:,} observations, kept {surviving:,} after bbox + filters")
    if not chunks:
        return pd.DataFrame(columns=keep_cols)
    return pd.concat(chunks, ignore_index=True)


def refine_with_europe_polygon(obs: pd.DataFrame, metadata_dir: Path, exclude_far_east: bool) -> pd.DataFrame:
    import geopandas as gpd
    from shapely.geometry import box

    shp = metadata_dir / "ne_110m_admin_0_countries" / "ne_110m_admin_0_countries.shp"
    print(f"[load] {shp}")
    countries = gpd.read_file(shp)

    cont_col = next((c for c in ["CONTINENT", "continent"] if c in countries.columns), None)
    if cont_col is None:
        raise RuntimeError(f"Natural Earth shapefile has no continent column: {countries.columns.tolist()}")

    europe = countries[countries[cont_col] == "Europe"].to_crs("EPSG:4326")
    europe_geom = europe.unary_union if hasattr(europe, "unary_union") else europe.geometry.union_all()

    if exclude_far_east:
        clip = box(-30, 30, 60, 80)
        europe_geom = europe_geom.intersection(clip)

    pts = gpd.GeoDataFrame(
        obs, geometry=gpd.points_from_xy(obs["longitude"], obs["latitude"]), crs="EPSG:4326"
    )
    print(f"[info] spatial join {len(pts):,} points against Europe polygon")
    inside = pts[pts.within(europe_geom)].drop(columns="geometry")
    print(f"[info] kept {len(inside):,} after polygon refine")
    return pd.DataFrame(inside)


def load_photos(metadata_dir: Path, observation_uuids: set[str]) -> pd.DataFrame:
    photos_path = metadata_dir / "photos.csv.gz"
    print(f"[load] {photos_path} (streaming, filtering to matching observation_uuids)")
    keep_cols = [
        "photo_uuid", "photo_id", "observation_uuid", "observer_id",
        "extension", "license", "width", "height", "position",
    ]
    chunks: list[pd.DataFrame] = []
    reader = pd.read_csv(
        photos_path, sep="\t", chunksize=1_000_000, usecols=keep_cols,
        dtype={
            "photo_uuid": "string", "photo_id": "Int64", "observation_uuid": "string",
            "observer_id": "Int64", "extension": "string", "license": "string",
            "width": "Int64", "height": "Int64", "position": "Int64",
        },
        low_memory=False,
    )
    for chunk in tqdm(reader, desc="photos", unit=" chunk"):
        chunk = chunk[chunk["observation_uuid"].isin(observation_uuids)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=keep_cols)
    return pd.concat(chunks, ignore_index=True)


def balance(df: pd.DataFrame, min_n: int, max_n: int, seed: int) -> pd.DataFrame:
    vc = df.groupby("taxon_id").size()
    keep_taxa = vc[vc >= min_n].index
    df = df[df["taxon_id"].isin(keep_taxa)]
    print(f"[info] {len(keep_taxa):,} species pass min_photos_per_species={min_n}")

    # Iterate groups directly to stay compatible with pandas 3.0
    # (groupby().apply() no longer accepts include_groups and drops the key column).
    sampled = [
        g.sample(n=max_n, random_state=seed) if len(g) > max_n else g
        for _, g in df.groupby("taxon_id", sort=False)
    ]
    return pd.concat(sampled, ignore_index=True)


def main() -> None:
    cfg = load_config()
    paths = project_paths(cfg)
    paths["filtered"].mkdir(parents=True, exist_ok=True)

    insects = build_insect_taxa(paths["metadata"], paths["filtered"])
    insect_ids = set(insects["taxon_id"].dropna().astype("int64").tolist())

    bbox = tuple(cfg["bbox"])  # (lat_min, lon_min, lat_max, lon_max)
    obs = filter_observations(paths["metadata"], insect_ids, bbox)

    if cfg.get("use_precise_europe_polygon", True):
        obs = refine_with_europe_polygon(obs, paths["metadata"], cfg.get("exclude_russian_far_east", True))

    obs_out = paths["filtered"] / "obs_europe_insects.parquet"
    obs.to_parquet(obs_out, index=False)
    print(f"[save] {obs_out} ({len(obs):,} rows)")

    photos = load_photos(paths["metadata"], set(obs["observation_uuid"]))
    print(f"[info] photos joined to filtered observations: {len(photos):,}")

    merged = photos.merge(
        obs[["observation_uuid", "taxon_id", "latitude", "longitude", "observed_on"]],
        on="observation_uuid", how="inner", suffixes=("", "_obs"),
    )
    merged = merged.merge(
        insects[["taxon_id", "name"]].rename(columns={"name": "scientific_name"}),
        on="taxon_id", how="inner",
    )
    merged["slug"] = merged["scientific_name"].fillna("").map(slugify)

    licenses = set(cfg["licenses_allowed"])
    before = len(merged)
    merged = merged[merged["license"].isin(licenses)]
    print(f"[info] license filter: {before:,} -> {len(merged):,}")

    merged = balance(merged, cfg["min_photos_per_species"], cfg["max_photos_per_species"], cfg["random_seed"])

    out_cols = [
        "photo_id", "extension", "taxon_id", "scientific_name", "slug",
        "license", "observer_id", "observation_uuid",
        "latitude", "longitude", "observed_on",
    ]
    merged = merged[out_cols].dropna(subset=["photo_id", "extension"])
    out = paths["filtered"] / "photos_to_download.parquet"
    merged.to_parquet(out, index=False)
    print(f"[save] {out} ({len(merged):,} photos, {merged['taxon_id'].nunique():,} species)")

    est_bytes = len(merged) * SIZE_BYTES_ESTIMATE.get(cfg["size"], 25_000)
    est_gb = est_bytes / 1e9
    print(f"[info] estimated disk for size={cfg['size']}: {est_gb:.1f} GB")
    if est_gb > cfg["disk_budget_gb"]:
        print(f"[WARN] estimate ({est_gb:.1f} GB) exceeds disk_budget_gb={cfg['disk_budget_gb']} GB.")
        print(f"[WARN] script 3 will hard-stop at the budget; lower max_photos_per_species to plan ahead.")

    counts = merged.groupby(["taxon_id", "scientific_name"]).size().sort_values(ascending=False)
    print("\n--- Top 20 species by photo count ---")
    print(counts.head(20).to_string())
    print("\n--- Bottom 20 species by photo count ---")
    print(counts.tail(20).to_string())

    if cfg.get("delete_metadata_after_filter", False):
        print(f"\n[clean] removing {paths['metadata']}")
        shutil.rmtree(paths["metadata"], ignore_errors=True)


if __name__ == "__main__":
    main()
