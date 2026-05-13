#!/usr/bin/env python3
"""
prepare_clustering_matrices.py

Prepares NMF clustering input matrices from per-case .npz tissue cache files.

Output:
  normed/      - L1 pattern-normalised (polar_calcific + polar_fibrotic + unwrap_calcific)
                 Spatially binned to 10x60 grids

Pattern normalisation (L1): each case's spatial map is divided by its total
tissue volume so each row sums to 1. NMF then decomposes distribution patterns
only, independent of burden. Burden is retained as a separate covariate for
post-hoc clinical testing. Approach follows Alexandrov et al. Nature 2013 and
Brunet et al. PNAS 2004.

Spatial resolution rationale:
  Polar 10x60 (600 features): 10 radial bins (~2mm each from annulus to STJ),
  60 angular bins (20 per cusp) provides sufficient resolution to distinguish
  inter-cusp and intra-cusp patterns without overfitting.
  Unwrap 10x60 (600 features): 10 longitudinal bins, 60 angular bins.
  Total 1800 features for ~1000 cases gives ~1.8:1 feature-to-case ratio.

Full resolution visualisation grids (20x120 polar, 60x120 unwrap) are retained
in the tissue feature cache for plotting but are not used for clustering.

Channels:
  polar_calcific.csv  : n x 600  leaflet calcific distribution
  polar_fibrotic.csv  : n x 600  leaflet fibrotic distribution
  unwrap_calcific.csv : n x 600  root calcific distribution

Usage:
    python prepare_clustering_matrices.py
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger(__name__)

CACHE_DIR    = Path("results/tissue_features/.cache")
FEATURES_CSV = Path("results/phenotyping/ct_features.csv")
OUTPUT_DIR   = Path("results/phenotyping/clustering_input")

# Full resolution (from tissue feature cache, used for visualisation only)
FULL_N_RADIAL       = 20
FULL_N_ANGULAR      = 120
FULL_N_LONGITUDINAL = 60

# Clustering resolution (spatially binned for NMF)
N_RADIAL       = 10
N_ANGULAR      = 60
N_LONGITUDINAL = 10


def load_annular_areas(features_csv: Path):
    df = pd.read_csv(features_csv)
    df["annular_area_mm2"] = df["annular_area_index"] * df["bsa"]
    missing = df["annular_area_mm2"].isna().sum()
    if missing > 0:
        log.warning(f"  {missing} cases have missing annular area and will be excluded")
    return (
        dict(zip(df["case_name"], df["annular_area_mm2"])),
        dict(zip(df["case_name"], df["valve_type"]))
    )


def load_npz(path: Path):
    try:
        d = np.load(str(path), allow_pickle=False)
        return {
            "case_name":       str(d["case_name"][0]),
            "valve_type":      str(d["valve_type"][0]),
            "polar_calcific":  d["polar_calcific"].astype(np.float64),
            "polar_fibrotic":  d["polar_fibrotic"].astype(np.float64),
            "unwrap_calcific": d["unwrap_calcific"].astype(np.float64),
        }
    except Exception as e:
        log.warning(f"  Failed to load {path.name}: {e}")
        return None


def spatial_downsample(grid: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    """Sum-pool adjacent bins to produce a coarser grid. Preserves total volume."""
    in_rows, in_cols = grid.shape
    row_factor = in_rows // target_rows
    col_factor = in_cols // target_cols
    trimmed = grid[:target_rows * row_factor, :target_cols * col_factor]
    return (trimmed
            .reshape(target_rows, row_factor, target_cols, col_factor)
            .sum(axis=(1, 3)))


def l1_normalise(vec: np.ndarray) -> np.ndarray:
    """
    L1 normalise a 1D array so it sums to 1.
    If sum is zero (no tissue), return zeros.
    """
    total = vec.sum()
    if total < 1e-12:
        return vec * 0.0
    return vec / total


def build_matrices(cache_dir: Path, ann_areas: dict, valve_types: dict) -> dict:
    npz_files = sorted(cache_dir.glob("*.npz"))
    log.info(f"Found {len(npz_files)} .npz files in {cache_dir}")

    channels = {ch: {"rows": [], "cases": [], "vtypes": []}
                for ch in ("polar_calcific", "polar_fibrotic", "unwrap_calcific")}

    n_loaded  = 0
    n_no_area = 0

    for fpath in npz_files:
        r = load_npz(fpath)
        if r is None:
            continue

        case   = r["case_name"]
        aa_mm2 = ann_areas.get(case, np.nan)
        if np.isnan(aa_mm2) or aa_mm2 <= 0:
            n_no_area += 1
            continue

        aa_cm2   = aa_mm2 / 100.0
        n_loaded += 1
        vtype    = valve_types.get(case, r["valve_type"])

        for ch in ("polar_calcific", "polar_fibrotic", "unwrap_calcific"):
            raw = r[ch]
            is_polar = "polar" in ch

            # Area-index (mm3/cm2)
            indexed = raw / aa_cm2

            # Spatial downsample from full to clustering resolution
            t_rows = N_RADIAL if is_polar else N_LONGITUDINAL
            indexed = spatial_downsample(indexed, t_rows, N_ANGULAR)

            # L1 pattern normalisation
            flat = indexed.ravel()
            flat_normed = l1_normalise(flat)

            channels[ch]["rows"].append(flat_normed)
            channels[ch]["cases"].append(case)
            channels[ch]["vtypes"].append(vtype)

    log.info(f"Loaded {n_loaded} cases ({n_no_area} excluded: no annular area)")

    result = {}
    for ch, data in channels.items():
        if not data["cases"]:
            log.warning(f"  {ch}: no valid cases")
            continue
        mat = np.stack(data["rows"], axis=0)
        result[ch] = {"mat": mat, "cases": data["cases"], "vtypes": data["vtypes"]}
        log.info(f"  {ch}: {mat.shape[0]} x {mat.shape[1]} [L1 NORMED]")

    return result


def write_csv(mat: np.ndarray, cases: list, path: Path):
    df = pd.DataFrame(mat, columns=[f"f{i}" for i in range(mat.shape[1])])
    df.insert(0, "case_name", cases)
    df.to_csv(path, index=False, float_format="%.8f")
    log.info(f"    Wrote {path.name}  ({mat.shape[0]} x {mat.shape[1]})")


def write_outputs(matrices: dict, output_dir: Path):
    out_dir = output_dir / "normed"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases  = matrices["polar_calcific"]["cases"]
    vtypes = matrices["polar_calcific"]["vtypes"]

    for ch in ("polar_calcific", "polar_fibrotic", "unwrap_calcific"):
        if ch not in matrices:
            log.warning(f"  {ch}: missing, skipping")
            continue
        write_csv(matrices[ch]["mat"], cases, out_dir / f"{ch}.csv")

    index_df = pd.DataFrame({"case_name": cases, "valve_type": vtypes})
    index_df.to_csv(output_dir / "case_index.csv", index=False)
    log.info(f"    Wrote case_index.csv  ({len(index_df)} cases)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir",    default=str(CACHE_DIR))
    parser.add_argument("--features_csv", default=str(FEATURES_CSV))
    parser.add_argument("--output_dir",   default=str(OUTPUT_DIR))
    args = parser.parse_args()

    cache_dir    = Path(args.cache_dir)
    features_csv = Path(args.features_csv)
    output_dir   = Path(args.output_dir)

    log.info(f"Cache:  {cache_dir}")
    log.info(f"Output: {output_dir}")
    log.info(f"Resolution: polar {N_RADIAL}x{N_ANGULAR}, unwrap {N_LONGITUDINAL}x{N_ANGULAR}")
    log.info(f"Normalisation: L1 per-case per-channel")

    ann_areas, valve_types = load_annular_areas(features_csv)
    log.info(f"Annular areas loaded for {len(ann_areas)} cases")

    matrices = build_matrices(cache_dir, ann_areas, valve_types)
    if not matrices:
        log.error("No valid matrices produced")
        return

    write_outputs(matrices, output_dir)
    log.info("\nDone.")


if __name__ == "__main__":
    main()