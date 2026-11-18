# Methods and Code

This file maps each section of the manuscript Methods to the scripts and
settings that produced the published results. Constants are defined at the
top of each script.

## CT processing and annotation

A reference point was placed at the aortic valve on axial CT, and a cubic
region centered on that point was cropped and segmented with TotalSegmentator
(`heartchambers_highres` task). The cropping and TotalSegmentator steps are
run before this pipeline and are not included here.

`annotation/aortic_segmentator.py` is the interactive multiplanar annotation
tool. For each case it records the cusp nadirs, commissural points, central
coaptation point and STJ plane in `metadata.json`, and writes the overall,
LVOT and per-cusp masks. The LVOT plane is placed 4 mm below the annular
plane along its normal.

## Tissue classification

`annotation/derived_masks.py`, function `classify_tissue_bp_calibrated`.

- The CT is smoothed with a 3-voxel median filter
  (`BLOOD_POOL_MEDIAN_KERNEL = 3`).
- Blood-pool HU values are sampled within the aortic root mask (root without
  LVOT) and limited to 45 to 850 HU.
- Calcific tissue is defined as voxels above the blood-pool mean + 3 SD.
- Fibrotic tissue is defined as voxels between 45 HU
  (`FIBROTIC_FLOOR_HU = 45`) and the blood-pool mean − 3 SD.
- The wall mask is the outer 1.5 mm shell of the root mask
  (`EROSION_MM = 1.5`).
- For the wall projection, calcific tissue is also identified within the
  overall mask (root plus LVOT) dilated by 2 mm
  (`CALCIUM_PROXIMITY_MM = 2.0`), to include wall-adherent calcium.

## Polar and cylindrical wall projections

`features/tissue_features.py`.

- Polar projection (`project_mask_to_polar`). Calcific and fibrotic voxels in
  the root mask, with the 1.5 mm wall shell removed, are projected onto the
  annular plane. Radius is measured from the center of the annular plane and
  extends to 1.3 times the annular radius (`RADIAL_EXTENT = 1.3`). Each valve
  is rotated so that the non-coronary cusp is at the top. The native grid is
  20 radial × 120 angular bins.
- Cylindrical wall projection (`unwrap_to_cylindrical`). Calcific voxels in
  the 2 mm dilated overall mask are projected onto a grid that runs from the
  LVOT plane to the STJ plane. The native grid is 60 longitudinal × 120
  angular bins.
- Tissue volumes are saved in mm³ and indexed to annular area in
  `features/extract_features.py`.

## Clustering

`clustering/prepare_clustering_matrices.py` downsamples each projection to a
10 × 60 grid (600 bins) and L1-normalizes each patient's map so that it sums
to 1. This gives three matrices (polar calcific, polar fibrotic and wall
calcific).

`clustering/clustering.R`:

- NMF is applied to each matrix separately (R `NMF` package, Brunet method,
  30 random initializations per rank). The number of components per matrix
  was selected from ranks 2 to 6 as the rank at which the cophenetic
  correlation coefficient plateaued before declining at higher ranks, which
  gave 2 components for each matrix. The script computes the rank
  diagnostics for ranks 2 to 6 and uses the selected ranks
  (`--fixed-ranks pc=2,pf=2,uw=2`).
- The per-patient NMF weights are scaled to the range 0 to 1, weighted by the
  variance explained within each matrix, and combined.
- Consensus clustering uses `ConsensusClusterPlus` with Partition Around
  Medoids, Pearson distance, 500 resamples and an item-resampling fraction of
  0.85.
- Mean silhouette width, cophenetic correlation and the proportion of
  ambiguous clustering are reported for each k in `k_selection_table.csv`.
  In the manuscript cohort, k=3 had the highest mean silhouette width across
  k=2 to 5, and the script uses k=3 by default (`--target-k`).
- Per-cluster stability is assessed by bootstrap Jaccard similarity over case
  resampling.

Bicuspid valves are excluded from clustering and analyzed separately.

## Outputs

- `cluster_assignments_<version>_k3.csv` gives the cluster for each case
  (1 = leaflet-fibrotic, 2 = leaflet-calcific, 3 = periannular-calcific).
- `validation_metrics.csv` gives the silhouette width, cophenetic correlation
  and proportion of ambiguous clustering.
- `jaccard_per_cluster.csv` gives the bootstrap Jaccard similarity for each
  cluster.

Random seeds are fixed in the clustering script.
