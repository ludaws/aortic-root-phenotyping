# Aortic Root Fibrocalcific Phenotyping

Source code accompanying:

> Dawson LP, Vekstein A, Huckaby L, et al. **Fibrocalcific tissue
> distribution phenotypes in tricuspid and bicuspid aortic stenosis.**
> *JACC: Cardiovascular Imaging* (under review).

This repository contains the full image-processing, tissue-classification, and
unsupervised-clustering pipeline used to derive three reproducible fibrocalcific
spatial phenotypes from pre-procedural cardiac CT in 1,129 patients undergoing
TAVR.

## Pipeline overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `annotation/aortic_segmentator.py` | Interactive MPR tool for cusp-nadir, commissure, annular, STJ, and LVOT landmark placement. Outputs anatomical masks (overall, LVOT, per-cusp) and `metadata.json`. |
| 2 | `annotation/derived_masks.py` | Generates wall, valve, blood-pool, calcific, and fibrotic masks using scan-specific blood-pool HU calibration. |
| 3 | `features/extract_features.py` | Extracts geometric features of the root, annulus, LVOT, and STJ. |
| 4 | `features/tissue_features.py` | Computes polar bullseye and cylindrical unwrap tissue distribution projections. |
| 5 | `clustering/prepare_clustering_matrices.py` | Builds L1-normalised feature matrices for NMF clustering. |
| 6 | `clustering/clustering.R` | NMF decomposition + ConsensusClusterPlus k-medoids consensus clustering. |

For a section-by-section mapping of these scripts to the published Methods, see
[`METHODS.md`](METHODS.md).

## Requirements

**Python** (see `requirements.txt` for full list):

- Python 3.10+
- numpy, scipy, pandas
- SimpleITK, pydicom, scikit-image
- matplotlib, napari (segmentator GUI)

**R**:

- R 4.4+
- `NMF`, `ConsensusClusterPlus`, `cluster`, `doParallel`,
  `dplyr`, `readr`, `jsonlite`

**External**:

- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
  for initial heart-chamber segmentation
  (run with `heartchambers_highres` task before step 1).

## Quick start

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run TotalSegmentator on each CT case (heartchambers_highres task)
# ... external step ...

# 1. Interactive landmark annotation (per case)
python annotation/aortic_segmentator.py \
    --data_dir <path/to/dicom_or_npz> \
    --totalseg_dir <path/to/totalseg_outputs>

# 2. Generate derived tissue and structural masks
python annotation/derived_masks.py --all

# 3-4. Geometric and spatial tissue features
python features/extract_features.py --all
python features/tissue_features.py --all

# 5. Build clustering input matrices
python clustering/prepare_clustering_matrices.py

# 6. NMF + consensus clustering (R)
Rscript clustering/clustering.R
```

## Tissue thresholds

Calcific and fibrotic tissue are classified using scan-specific blood-pool
calibration:

- Blood pool sampled within the overall root/LVOT mask, clipped to
  the 45–850 HU range
- Calcific threshold: blood-pool mean + 3 SD
- Fibrotic threshold: between 45 HU and blood-pool mean − 3 SD

This follows Lembo et al. *JACC Cardiovasc Imaging* 2024 and Grodecki et al.
*Radiology* 2024, both of which validate the approach against histology.

## Data

Patient-level clinical and imaging data are not publicly available due to
institutional data-use agreements. Derived per-case feature files
(non-identifiable) may be available from the corresponding author on reasonable
request.

## Citation

If you use this code, please cite the manuscript above.

## Licence

MIT — see [`LICENSE`](LICENSE).

## Contact

Luke P. Dawson — `lukepdawson1@gmail.com`
