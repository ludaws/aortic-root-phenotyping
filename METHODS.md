# Methods–Code Mapping

This document maps each section of the manuscript Methods to the specific
scripts and functions used to produce the published results.

## CT Image Assessment & Processing

Manuscript text:
> First, a reference point was manually placed at the aortic valve on axial CT
> images and a 140mm × 140mm region of interest was cropped. These were
> processed using TotalSegmentator to generate automated segmentations of the
> aorta and LV. A custom python script was used to manually review the CT
> images and aorta/LV segmentations with multi-planar reconstruction, and the
> following manual annotations were made: cusp nadir points, central
> coaptation point and commissural points, and the STJ plane. The LVOT plane
> was defined as 4mm below the aortic annulus.

Implementation:

- TotalSegmentator is an external dependency, run with `heartchambers_highres`.
- `annotation/aortic_segmentator.py` provides the interactive MPR annotator and
  writes per-case anatomical masks plus `metadata.json` containing the
  annular, STJ, and LVOT plane definitions, cusp nadirs, commissural points,
  and central coaptation point.
- The LVOT plane is computed as 4 mm below the annular plane along its normal
  (see `metadata.json["lvot_plane"]`).

## Fibrocalcific Tissue Feature Definitions

Manuscript text:
> Calcific and fibrotic tissue volumes were quantified using scan-specific
> blood pool calibration previously validated against histology. The mean and
> standard deviation (SD) of Hounsfield Unit (HU) values for the blood pool
> within the aortic root mask were calculated (limited to an HU range of
> 45–850), with calcific tissue defined as voxels exceeding three SDs above
> the mean and fibrotic tissue as voxels between three SDs below the mean and
> 45 HU.

Implementation: `annotation/derived_masks.py`. Key constants:

- `FIBROTIC_FLOOR_HU = 45` — lower HU bound for fibrotic tissue.
- Calcific and fibrotic thresholds are derived per-case from the
  blood-pool distribution within `overall_mask`, clipped to 45–850 HU
  (see `classify_tissue` function).

## Polar Bullseye and Cylindrical Unwrap Projections

Manuscript text:
> For the polar bullseye projection, calcific and fibrotic voxels within the
> root mask (eroded by 1.5mm to exclude wall tissue) were projected onto the
> annular plane. The projection was mapped to a 10×60 spatial grid... For the
> cylindrical wall projection, the overall mask (root plus LVOT, dilated 2mm
> to capture wall-adherent calcium) was projected onto a rectangular grid
> bounded by the LVOT and STJ planes, and mapped to a 10×60 spatial grid.

Implementation: `features/tissue_features.py`.

- Polar projection: `project_mask_to_polar`. Native grid is 20×120 (see
  `N_RADIAL_BINS`, `N_ANGULAR_BINS`); downsampled to 10×60 for clustering in
  `clustering/prepare_clustering_matrices.py`.
- Unwrap projection: `unwrap_to_cylindrical`. Native grid 60×120;
  downsampled to 10×60 for clustering.
- Erosion uses `wall_mask`, defined as the 1.5 mm wall shell in
  `derived_masks.py` (`EROSION_MM = 1.5`).
- Calcific dilation uses `calcific_mask_dilated`, generated within a 2 mm
  expansion of `overall_mask` (`CALCIUM_PROXIMITY_MM = 2.0`).

## Geometric Feature Extraction

Manuscript text:
> Geometric features describing the root, annulus, LVOT, and STJ were derived
> from the annotated masks and used to assess correlations with tissue
> parameters.

Implementation: `features/extract_features.py`. Produces annular area, root
volume, LVOT and STJ dimensions, plane tilts, and per-cusp rotation angles.

## Cluster Phenotyping

Manuscript text:
> Each of three 600-bin matrices (polar calcific, polar fibrotic, and wall
> calcific) were L1-normalized so that each patient's map summed to 1.
> NMF was applied to each matrix independently for dimensionality reduction.
> The resulting per-patient NMF weights were combined and used as input for
> consensus clustering. Consensus clustering was performed using Partition
> Around Medoids with Pearson distance for k=2 to 6 cluster groups.

Implementation:

- `clustering/prepare_clustering_matrices.py` builds the L1-normalised
  10×60 = 600-feature matrices from the cached projections.
- `clustering/clustering.R`:
  - NMF per channel (R `NMF` package, Brunet method).
  - Per-channel H matrices range-normalised, weighted by within-channel
    explained variance, and concatenated to a per-case embedding.
  - `ConsensusClusterPlus` with PAM clustering, Pearson distance,
    500 resamples, item-resampling fraction 0.85, k = 2…6.
  - Stability assessed by cophenetic correlation of the consensus matrix,
    mean silhouette width, proportion of ambiguous clustering (PAC), and
    per-cluster bootstrap Jaccard similarity (Hennig 2007).

## Outputs

- `cluster_assignments_<version>_k3.csv` — final per-case cluster
  assignments (1 = leaflet-fibrotic, 2 = leaflet-calcific reference,
  3 = periannular-calcific).
- `validation_metrics.csv` — cophenetic correlation, silhouette, PAC, ARI.
- `jaccard_per_cluster.csv` — bootstrap stability per cluster.

## Reproducibility notes

- Code uses fixed random seeds where possible (`consensus_seeds = 42, 123,
  456`) and reports across multiple seeds.
- Bicuspid cases are excluded from the tricuspid clustering and analysed
  separately as described in the manuscript.
- All thresholds (HU, erosion, dilation, grid resolution) are defined as
  named constants at the top of each script.
