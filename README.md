# Aortic Root Fibrocalcific Phenotyping

Source code accompanying:

> Dawson LP, et al. **Fibrocalcific tissue distribution phenotypes in severe
> aortic stenosis.**

This repository contains the image-processing, tissue-classification and
unsupervised-clustering code used to derive three fibrocalcific tissue
distribution phenotypes from pre-TAVR cardiac CT in 1,014 patients with
tricuspid aortic stenosis. Patients with bicuspid valves (n=115) were
processed with the same pipeline and analyzed separately.

## Pipeline overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `annotation/aortic_segmentator.py` | Interactive multiplanar annotation of cusp nadirs, commissures, central coaptation point and the STJ plane. Writes the overall, LVOT and per-cusp masks and `metadata.json`. |
| 2 | `annotation/derived_masks.py` | Generates wall, valve, blood-pool, calcific and fibrotic masks using scan-specific blood-pool calibration. |
| 3 | `features/extract_features.py` | Extracts geometric features of the root, annulus, LVOT and STJ. |
| 4 | `features/tissue_features.py` | Computes the polar and cylindrical wall projections of calcific and fibrotic tissue. |
| 5 | `clustering/prepare_clustering_matrices.py` | Builds the L1-normalized feature matrices used for clustering. |
| 6 | `clustering/clustering.R` | NMF of each matrix followed by consensus clustering. |

[`METHODS.md`](METHODS.md) maps each section of the manuscript Methods to the
scripts and settings used.

## Requirements

Python 3.10 or later (see `requirements.txt`), including numpy, scipy,
pandas, SimpleITK, pydicom, scikit-image, matplotlib and napari.

R 4.4 or later, with the `NMF`, `ConsensusClusterPlus`, `cluster`,
`doParallel`, `dplyr`, `readr` and `jsonlite` packages.

[TotalSegmentator](https://github.com/wasserth/TotalSegmentator), run with
the `heartchambers_highres` task on each CT before step 1.

## Quick start

```bash
pip install -r requirements.txt

# TotalSegmentator (heartchambers_highres) is run on each CT before step 1.

# 1. Landmark annotation, one case at a time
python annotation/aortic_segmentator.py \
    --data_dir <path/to/dicom_or_npz> \
    --totalseg_dir <path/to/totalseg_outputs>

# 2. Derived tissue and structural masks
python annotation/derived_masks.py --all

# 3. Geometric features (needs a cohort CSV with BSA and valve type)
python features/extract_features.py \
    --seg_dir data/segmentations \
    --csv <path/to/cohort.csv>

# 4. Polar and wall projections
python features/tissue_features.py --all

# 5. Clustering input matrices
python clustering/prepare_clustering_matrices.py

# 6. NMF and consensus clustering
Rscript clustering/clustering.R
```

## Tissue thresholds

Calcific and fibrotic tissue are classified with scan-specific blood-pool
calibration, following Lembo et al. (JACC Cardiovasc Imaging 2024) and
Grodecki et al. (Radiology 2024).

- The blood pool is sampled within the aortic root mask after a 3-voxel
  median filter, limited to 45 to 850 HU.
- Calcific tissue is defined as voxels above the blood-pool mean + 3 SD.
- Fibrotic tissue is defined as voxels between 45 HU and the blood-pool
  mean − 3 SD.

## Data

Patient-level clinical and imaging data are not publicly available because of
institutional data-use agreements. Derived per-case feature files, which
contain no identifiable information, may be available from the corresponding
author on reasonable request.

## Citation

Please cite the manuscript above if you use this code.

## Licence

MIT. See [`LICENSE`](LICENSE).

## Contact

Luke P. Dawson (`lukepdawson1@gmail.com`)
