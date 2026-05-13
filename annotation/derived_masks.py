#!/usr/bin/env python3
"""
Derived Mask Generator.

Generates secondary masks from the interactive annotator output and the
raw DICOM volume:

  wall_mask.nii.gz              - 1.5 mm outer shell of overall_mask
                                  (plane-aware EDT; annular and STJ faces
                                  excluded as erosion surfaces)
  valve_mask.nii.gz             - eroded interior intersected with cusp masks
  annular_mask.nii.gz           - 2 mm thick ring at the annular plane
  blood_pool_mask.nii.gz        - blood pool sampled within overall_mask (QC)
  calcific_mask.nii.gz          - blood-pool-calibrated calcific tissue
                                  within overall_mask
  fibrotic_mask.nii.gz          - blood-pool-calibrated fibrotic tissue
                                  within overall_mask
  calcific_mask_dilated.nii.gz  - calcific tissue within 2 mm-dilated
                                  overall_mask (captures wall-adherent calcium)
  calcium_mask.nii.gz           - fixed HU > 850 within dilated overall_mask
                                  (for nodule morphometry)

Tissue classification uses scan-specific blood-pool calibration following
Lembo et al. (JACC Cardiovasc Imaging 2024;17:1351) and Grodecki et al.
(Radiology 2024;312:e240229):
  - Blood pool sampled within overall_mask, clipped to 45-850 HU.
  - Blood-pool mean and SD computed from the clipped distribution.
  - Fibrotic threshold = BP mean - 3*SD (down to 45 HU floor).
  - Calcific threshold = BP mean + 3*SD.

Per-case thresholds are written to tissue_params.json.

Usage:
    python derived_masks.py --all
    python derived_masks.py --case_name <case>
    python derived_masks.py --case_name <case> --check
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pydicom
from scipy import ndimage
from scipy.ndimage import median_filter
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults (overridable via CLI)
# ---------------------------------------------------------------------------
SEGMENTATION_DIR = Path("data/segmentations")
DICOM_DIR = Path("data/raw")
TOTALSEG_DIR = Path("data/totalseg_outputs")

CALCIUM_HU_THRESHOLD = 850       # Conservative fixed threshold for nodule morphometry
CALCIUM_PROXIMITY_MM = 2.0       # Dilation from overall_mask boundary for calcium search
ANNULAR_RING_THICKNESS_MM = 2.0  # Thickness of annular ring mask
EROSION_MM = 1.5                 # Shell thickness for wall/valve split
FIBROTIC_FLOOR_HU = 45           # Lower bound for fibrotic tissue (Grodecki 2024 Appendix S1)
BLOOD_POOL_MEDIAN_KERNEL = 3     # Median filter kernel size for blood pool smoothing


# ===================================================================
# I/O helpers
# ===================================================================

def load_dicom_volume(case_folder: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load DICOM volume and return (volume_HU, spacing_zyx, origin_xyz)."""
    dicom_files = []
    for f in case_folder.rglob('*'):
        if f.is_file() and not f.name.startswith('.'):
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
                if hasattr(ds, 'Modality') and ds.Modality == 'CT':
                    dicom_files.append(f)
            except Exception:
                continue

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {case_folder}")

    slices = []
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(f)
            if hasattr(ds, 'ImagePositionPatient') and hasattr(ds, 'pixel_array'):
                slices.append(ds)
        except Exception:
            continue

    if len(slices) < 10:
        raise ValueError(f"Too few valid slices: {len(slices)}")

    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    ds0 = slices[0]
    pixel_spacing = [float(x) for x in ds0.PixelSpacing]
    slice_thickness = abs(
        float(slices[1].ImagePositionPatient[2])
        - float(slices[0].ImagePositionPatient[2])
    )

    volume = np.stack([s.pixel_array for s in slices], axis=0).astype(np.int16)
    slope = float(getattr(ds0, 'RescaleSlope', 1))
    intercept = float(getattr(ds0, 'RescaleIntercept', 0))
    if slope != 1 or intercept != 0:
        volume = (volume.astype(np.float32) * slope + intercept).astype(np.int16)

    spacing_zyx = np.array([slice_thickness, pixel_spacing[1], pixel_spacing[0]])
    origin_xyz = np.array([float(x) for x in ds0.ImagePositionPatient])

    return volume, spacing_zyx, origin_xyz


def load_sitk_mask(path: Path) -> Tuple[np.ndarray, list, list]:
    """Load NIfTI mask. Returns (array, spacing_xyz, origin_xyz) in SimpleITK order."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    spacing = list(img.GetSpacing())
    origin = list(img.GetOrigin())
    return arr.astype(np.uint8), spacing, origin


def save_sitk_mask(data: np.ndarray, spacing: list, origin: list, path: Path):
    """Save mask with SimpleITK, matching aortic_segmentator conventions."""
    img = sitk.GetImageFromArray(data.astype(np.uint8))
    img.SetSpacing([float(s) for s in spacing])
    img.SetOrigin([float(o) for o in origin])
    sitk.WriteImage(img, str(path))


# ===================================================================
# Mask generation functions
# ===================================================================

def _build_coordinate_grids(
    shape: tuple,
    spacing_zyx: np.ndarray,
    origin_xyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build physical coordinate grids (x, y, z) for a volume.

    Returns (xx, yy, zz) arrays in shape (nz, ny, nx) matching mask indexing.
    """
    sx, sy, sz = spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]
    ox, oy, oz = origin_xyz[0], origin_xyz[1], origin_xyz[2]

    x_coords = ox + np.arange(shape[2]) * sx
    y_coords = oy + np.arange(shape[1]) * sy
    z_coords = oz + np.arange(shape[0]) * sz

    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    xx = np.transpose(xx, (2, 1, 0))
    yy = np.transpose(yy, (2, 1, 0))
    zz = np.transpose(zz, (2, 1, 0))
    return xx, yy, zz


def _signed_plane_distance(
    xx: np.ndarray, yy: np.ndarray, zz: np.ndarray,
    plane: Dict,
) -> np.ndarray:
    """Signed distance from each voxel to a plane (positive = normal side)."""
    c = np.array(plane['center'])
    n = np.array(plane['normal'])
    n = n / np.linalg.norm(n)
    return (xx - c[0]) * n[0] + (yy - c[1]) * n[1] + (zz - c[2]) * n[2]


def generate_wall_valve_masks(
    root_mask: np.ndarray,
    cusp_masks: Dict[str, np.ndarray],
    spacing_xyz: list,
    metadata: Dict,
    origin_xyz: np.ndarray,
    erosion_mm: float = EROSION_MM,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate wall (lateral shell) and valve (interior leaflet) masks.

    Uses distance_transform_edt with physical spacing for accurate mm erosion.
    To avoid the EDT treating the open ends (annular/STJ planes) as surfaces,
    we pad the mask by extending it beyond both planes before computing the
    distance transform. This ensures the EDT only measures distance to the
    lateral (sinus) walls.

    Wall = root_mask voxels within erosion_mm of the lateral surface.
    Valve = interior voxels (distance >= erosion_mm) intersected with cusp union.

    Returns (wall_mask, valve_mask) as uint8 arrays.
    """
    from scipy.ndimage import distance_transform_edt

    spacing_zyx = np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]])
    root_binary = root_mask > 0

    annulus_plane = metadata.get('annulus_plane')
    stj_plane = metadata.get('stj_plane')

    if annulus_plane and stj_plane:
        # Build coordinate grids
        xx, yy, zz = _build_coordinate_grids(
            root_mask.shape, spacing_zyx, origin_xyz
        )

        # Signed distance to each plane
        # Normal points from annulus toward STJ (upward through root)
        # Voxels between planes: dist_annular >= 0 and dist_stj <= 0
        dist_annular = _signed_plane_distance(xx, yy, zz, annulus_plane)
        dist_stj = _signed_plane_distance(xx, yy, zz, stj_plane)

        # Pad: extend root_mask beyond both planes to eliminate end-face
        # artifacts in EDT. Use max z-projection of root_mask as the
        # lateral footprint, then fill all voxels beyond each plane that
        # fall within this footprint.
        footprint_yx = np.any(root_binary, axis=0)  # (ny, nx)

        beyond_annular = (dist_annular < 0) & footprint_yx[np.newaxis, :, :]
        beyond_stj = (dist_stj > 0) & footprint_yx[np.newaxis, :, :]

        padded = root_binary | beyond_annular | beyond_stj

        logger.info(
            f"  EDT padding: beyond annular {int(np.sum(beyond_annular))} vx, "
            f"beyond STJ {int(np.sum(beyond_stj))} vx"
        )
    else:
        logger.warning("  Missing plane metadata, EDT without end-padding")
        padded = root_binary

    # Distance transform in physical mm
    dist_map = distance_transform_edt(padded, sampling=spacing_zyx)

    # Wall and interior from the original root_mask only
    wall_mask = (root_binary & (dist_map < erosion_mm)).astype(np.uint8)
    interior = root_binary & (dist_map >= erosion_mm)

    cusp_union = np.zeros_like(root_mask, dtype=bool)
    for mask in cusp_masks.values():
        cusp_union |= (mask > 0)

    valve_mask = (interior & cusp_union).astype(np.uint8)

    root_voxels = int(np.sum(root_binary))
    logger.info(
        f"  Wall/valve: EDT erosion {erosion_mm}mm, "
        f"wall {int(np.sum(wall_mask))} vx ({int(np.sum(wall_mask)) / max(root_voxels, 1):.1%}), "
        f"valve {int(np.sum(valve_mask))} vx ({int(np.sum(valve_mask)) / max(root_voxels, 1):.1%}), "
        f"interior {int(np.sum(interior))} vx ({int(np.sum(interior)) / max(root_voxels, 1):.1%})"
    )
    return wall_mask, valve_mask


def generate_annular_mask(
    overall_mask: np.ndarray,
    metadata: Dict,
    spacing_zyx: np.ndarray,
    origin_xyz: np.ndarray,
    thickness_mm: float = ANNULAR_RING_THICKNESS_MM,
) -> np.ndarray:
    """Generate annular ring mask at outer edge of overall_mask at annular plane."""
    from scipy.ndimage import distance_transform_edt

    annular_mask = np.zeros_like(overall_mask, dtype=np.uint8)

    annulus_plane = metadata.get('annulus_plane')
    if not annulus_plane:
        logger.warning("  No annulus_plane in metadata")
        return annular_mask

    half_thickness = thickness_mm / 2.0

    xx, yy, zz = _build_coordinate_grids(
        overall_mask.shape, spacing_zyx, origin_xyz
    )
    dist_to_plane = np.abs(_signed_plane_distance(xx, yy, zz, annulus_plane))

    # Outer shell via EDT
    dist_map = distance_transform_edt(overall_mask > 0, sampling=spacing_zyx)
    outer_shell = (overall_mask > 0) & (dist_map < EROSION_MM)

    in_slab = dist_to_plane <= half_thickness
    annular_mask = (outer_shell & in_slab).astype(np.uint8)

    logger.info(
        f"  Annular ring: {int(np.sum(annular_mask))} vx "
        f"(shell {int(np.sum(outer_shell))}, slab {int(np.sum(in_slab & (overall_mask > 0)))})"
    )
    return annular_mask


def generate_calcium_mask(
    ct_volume: np.ndarray,
    overall_mask: np.ndarray,
    spacing_zyx: np.ndarray,
    hu_threshold: float = CALCIUM_HU_THRESHOLD,
    proximity_mm: float = CALCIUM_PROXIMITY_MM,
) -> np.ndarray:
    """Generate conservative calcium mask (fixed HU threshold) within dilated overall_mask."""
    from scipy.ndimage import distance_transform_edt

    # Dilate via EDT on inverted mask: distance from background to nearest foreground
    dist_outside = distance_transform_edt(~(overall_mask > 0), sampling=spacing_zyx)
    dilated_mask = (overall_mask > 0) | (dist_outside <= proximity_mm)

    calcium_mask = ((ct_volume > hu_threshold) & dilated_mask).astype(np.uint8)
    logger.info(f"  Calcium (HU>{hu_threshold}): {int(np.sum(calcium_mask))} vx")
    return calcium_mask


# ===================================================================
# Blood-pool-calibrated tissue classification (Dey 2009 / Grodecki 2024)
# ===================================================================

def classify_tissue_bp_calibrated(
    ct_volume: np.ndarray,
    overall_mask: np.ndarray,
    calibration_mask: np.ndarray,
    fibrotic_floor_hu: float = FIBROTIC_FLOOR_HU,
    calcium_clip_hu: float = 850.0,
    median_kernel: int = BLOOD_POOL_MEDIAN_KERNEL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Classify voxels into fibrotic, blood pool, and calcific tissue.

    Implements scan-specific blood pool calibration as described by
    Patel et al. (Eur Heart J Cardiovasc Imaging 2023;24:1653-1660) and
    Grodecki et al. (Radiology 2024;312:e240229):

      1. Extract voxels within calibration_mask (root_mask, aortic root without LVOT)
      2. Apply median filter to reduce noise
      3. Clip to [fibrotic_floor_hu, calcium_clip_hu] to isolate blood pool
      4. Compute blood pool mean and SD from clipped distribution
      5. Fibrotic threshold = mean - 3*SD (99.7th percentile lower bound)
      6. Calcific threshold = mean + 3*SD (99.7th percentile upper bound)

    Thresholds represent the 99.7th percentile bounds of the blood pool
    Gaussian, per Patel et al. 2023.

    Args:
        ct_volume: CT volume in HU (z, y, x)
        overall_mask: Binary mask for tissue classification output (root_mask)
        calibration_mask: Binary mask for blood pool sampling (root_mask)
        fibrotic_floor_hu: Lower bound clip for blood pool sampling (excludes air/artifact)
        calcium_clip_hu: Upper bound clip for blood pool sampling (excludes calcium)
        median_kernel: Kernel size for median filtering

    Returns:
        (fibrotic_mask, calcific_mask, blood_pool_mask, tissue_params)
    """
    mask_binary = overall_mask > 0
    cal_binary = calibration_mask > 0

    # Apply median filter to reduce noise (per Dey 2009)
    ct_filtered = median_filter(ct_volume.astype(np.float64), size=median_kernel)

    # Sample blood pool from calibration_mask (root_mask)
    hu_cal = ct_filtered[cal_binary]

    # Clip to [fibrotic_floor, calcium_clip] to isolate blood pool
    hu_clipped = hu_cal[(hu_cal >= fibrotic_floor_hu) & (hu_cal <= calcium_clip_hu)]

    # Blood pool mean and SD from clipped distribution
    bp_mean = float(np.mean(hu_clipped))
    bp_sd = float(np.std(hu_clipped))

    # Thresholds: 99.7th percentile of blood pool (mean +/- 3SD)
    fibrotic_upper = bp_mean - 3.0 * bp_sd
    calcific_lower = bp_mean + 3.0 * bp_sd

    # Enforce fibrotic floor
    fibrotic_upper = max(fibrotic_upper, fibrotic_floor_hu)

    # Cartlidge cross-check
    cartlidge_equation_upper = 41.46 + 0.42 * bp_mean

    tissue_params = {
        'method': 'bp_clipped_mean_3sd',
        'reference': 'Patel et al. Eur Heart J Cardiovasc Imaging 2023;24:1653-1660',
        'blood_pool': {
            'mean_hu': float(round(bp_mean, 1)),
            'sd_hu': float(round(bp_sd, 1)),
        },
        'clip_range': {
            'lower_hu': float(fibrotic_floor_hu),
            'upper_hu': float(calcium_clip_hu),
        },
        'thresholds': {
            'fibrotic_floor_hu': float(fibrotic_floor_hu),
            'fibrotic_upper_hu': float(round(fibrotic_upper, 1)),
            'calcific_lower_hu': float(round(calcific_lower, 1)),
        },
        'cartlidge_crosscheck': {
            'cartlidge_equation_upper': float(round(cartlidge_equation_upper, 1)),
        },
        'median_filter_kernel': median_kernel,
        'n_voxels_in_classification_mask': int(np.sum(mask_binary)),
        'n_voxels_in_calibration_mask': int(np.sum(cal_binary)),
        'n_voxels_clipped': int(len(hu_clipped)),
    }

    logger.info(f"  Blood pool calibration (clipped mean+/-3SD):")
    logger.info(f"    Calibration mask: {int(np.sum(cal_binary))} vx, clipped [{fibrotic_floor_hu}, {calcium_clip_hu}]: {len(hu_clipped)} vx")
    logger.info(f"    Blood pool: mean={bp_mean:.1f} HU, SD={bp_sd:.1f}")
    logger.info(
        f"    Thresholds: fibrotic=[{fibrotic_floor_hu}, {fibrotic_upper:.1f}] HU, "
        f"calcific=[{calcific_lower:.1f}, inf) HU"
    )
    logger.info(
        f"    Cartlidge cross-check: equation upper={cartlidge_equation_upper:.1f}"
    )

    # Build classification masks using UNFILTERED CT values
    fibrotic_mask = np.zeros_like(overall_mask, dtype=np.uint8)
    calcific_mask = np.zeros_like(overall_mask, dtype=np.uint8)
    blood_pool_mask = np.zeros_like(overall_mask, dtype=np.uint8)

    ct_float = ct_volume.astype(np.float64)

    fibrotic_mask[
        mask_binary
        & (ct_float >= fibrotic_floor_hu)
        & (ct_float < fibrotic_upper)
    ] = 1

    calcific_mask[
        mask_binary
        & (ct_float >= calcific_lower)
    ] = 1

    blood_pool_mask[
        mask_binary
        & (ct_float >= fibrotic_upper)
        & (ct_float < calcific_lower)
    ] = 1

    n_fibrotic = int(np.sum(fibrotic_mask))
    n_calcific = int(np.sum(calcific_mask))
    n_bp = int(np.sum(blood_pool_mask))
    n_total = int(np.sum(mask_binary))
    n_below_floor = int(np.sum(mask_binary & (ct_float < fibrotic_floor_hu)))

    tissue_params['voxel_counts'] = {
        'fibrotic': n_fibrotic,
        'blood_pool': n_bp,
        'calcific': n_calcific,
        'below_floor': n_below_floor,
        'total_in_mask': n_total,
    }

    logger.info(
        f"    Voxels: fibrotic={n_fibrotic}, blood_pool={n_bp}, "
        f"calcific={n_calcific}, below_floor={n_below_floor}"
    )

    return fibrotic_mask, calcific_mask, blood_pool_mask, tissue_params


# ===================================================================
# Case processing
# ===================================================================

def load_cusp_masks(seg_dir: Path, valve_type: str) -> Dict[str, np.ndarray]:
    """Load cusp masks based on valve type."""
    cusp_masks = {}
    if valve_type == 'type0':
        labels = ['A', 'P']
    else:
        labels = ['L', 'N', 'R']

    for label in labels:
        cusp_file = seg_dir / f'cusp_{label}_mask.nii.gz'
        if cusp_file.exists():
            arr, _, _ = load_sitk_mask(cusp_file)
            cusp_masks[label] = arr

    return cusp_masks


def process_case(case_name: str, overwrite: bool = False, tissue_only: bool = False,
                 wall_only: bool = False) -> bool:
    """Process a single case: generate all derived masks.

    wall_only: regenerate wall_mask.nii.gz only (no DICOM loading, no tissue
               classification). Uses root_mask + metadata + cusp masks only.
               Much faster than full processing — useful when changing EROSION_MM.
    """

    seg_dir = SEGMENTATION_DIR / case_name
    metadata_file = seg_dir / 'metadata.json'

    if not metadata_file.exists():
        logger.warning(f"No metadata.json for {case_name}")
        return False

    with open(metadata_file) as f:
        metadata = json.load(f)

    if metadata.get('ct_quality') == 'poor':
        logger.info(f"Skipping poor quality: {case_name}")
        return False

    if not metadata.get('annulus_set'):
        logger.info(f"Skipping incomplete (no annulus): {case_name}")
        return False

    # Check if all outputs exist already
    output_files = [
        'root_mask.nii.gz', 'wall_mask.nii.gz', 'valve_mask.nii.gz',
        'annular_mask.nii.gz', 'fibrotic_mask.nii.gz', 'calcific_mask.nii.gz',
        'calcific_mask_dilated.nii.gz', 'blood_pool_mask.nii.gz',
        'calcium_mask.nii.gz', 'calcium_mask_dilated.nii.gz', 'tissue_params.json',
    ]
    if not overwrite and not tissue_only and not wall_only and all((seg_dir / f).exists() for f in output_files):
        return False  # All exist, skip

    # --- Wall-only fast path (no DICOM needed) ---
    if wall_only:
        root_mask_file = seg_dir / 'root_mask.nii.gz'
        if not root_mask_file.exists():
            logger.warning(f"  wall_only: no root_mask.nii.gz for {case_name}, skipping")
            return False

        if not overwrite and (seg_dir / 'wall_mask.nii.gz').exists():
            return False

        root_mask, mask_spacing, mask_origin = load_sitk_mask(root_mask_file)

        crop_info_file = TOTALSEG_DIR / case_name / 'crop_info.json'
        if not crop_info_file.exists():
            logger.warning(f"  wall_only: no crop_info.json for {case_name}, skipping")
            return False
        with open(crop_info_file) as f:
            crop_info = json.load(f)
        origin_xyz = np.array(crop_info['cropped_origin'])

        valve_type = metadata.get('valve_type', 'tricuspid')
        cusp_masks = load_cusp_masks(seg_dir, valve_type)

        wall_mask, _ = generate_wall_valve_masks(
            root_mask, cusp_masks, mask_spacing, metadata, origin_xyz,
            erosion_mm=EROSION_MM,
        )
        save_sitk_mask(wall_mask, mask_spacing, mask_origin, seg_dir / 'wall_mask.nii.gz')
        logger.info(f"  wall_only: saved wall_mask ({int(np.sum(wall_mask))} vx, erosion={EROSION_MM}mm)")
        return True

    # --- Load inputs ---

    overall_mask_file = seg_dir / 'overall_mask.nii.gz'
    if not overall_mask_file.exists():
        logger.warning(f"No overall_mask.nii.gz for {case_name}")
        return False

    overall_mask, mask_spacing, mask_origin = load_sitk_mask(overall_mask_file)

    valve_type = metadata.get('valve_type', 'tricuspid')
    cusp_masks = load_cusp_masks(seg_dir, valve_type)
    if len(cusp_masks) == 0 and not tissue_only:
        logger.warning(f"No cusp masks found for {case_name} (valve_type={valve_type})")
        return False

    crop_info_file = TOTALSEG_DIR / case_name / 'crop_info.json'
    if not crop_info_file.exists():
        logger.warning(f"No crop_info.json for {case_name}")
        return False

    with open(crop_info_file) as f:
        crop_info = json.load(f)

    dicom_folder = DICOM_DIR / case_name
    if not dicom_folder.exists():
        logger.warning(f"No DICOM folder for {case_name}")
        return False

    try:
        ct_volume, ct_spacing, ct_origin = load_dicom_volume(dicom_folder)
    except Exception as e:
        logger.error(f"Failed to load DICOM for {case_name}: {e}")
        return False

    # Crop CT to match mask
    starts = crop_info['crop_starts']
    ends = crop_info['crop_ends']
    ct_cropped = ct_volume[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]

    # Spacing in (z, y, x) for scipy; crop_info spacing is SimpleITK (x, y, z)
    sitk_spacing = crop_info['spacing']
    spacing_zyx = np.array([sitk_spacing[2], sitk_spacing[1], sitk_spacing[0]])
    origin_xyz = np.array(crop_info['cropped_origin'])

    # Shape alignment
    if ct_cropped.shape != overall_mask.shape:
        logger.warning(
            f"Shape mismatch: CT {ct_cropped.shape} vs mask {overall_mask.shape}"
        )
        min_shape = [
            min(ct_cropped.shape[i], overall_mask.shape[i]) for i in range(3)
        ]
        ct_cropped = ct_cropped[:min_shape[0], :min_shape[1], :min_shape[2]]
        overall_mask = overall_mask[:min_shape[0], :min_shape[1], :min_shape[2]]
        # Trim cusp masks too
        for label in cusp_masks:
            cusp_masks[label] = cusp_masks[label][
                :min_shape[0], :min_shape[1], :min_shape[2]
            ]

    logger.info(
        f"Processing {case_name} (valve_type={valve_type}, "
        f"{len(cusp_masks)} cusps)..."
    )

    # --- Generate masks ---

    # 1. Root mask = overall minus LVOT
    lvot_mask_file = seg_dir / 'lvot_mask.nii.gz'
    if lvot_mask_file.exists():
        lvot_mask, _, _ = load_sitk_mask(lvot_mask_file)
        if lvot_mask.shape != overall_mask.shape:
            min_shape = [min(lvot_mask.shape[i], overall_mask.shape[i]) for i in range(3)]
            lvot_mask = lvot_mask[:min_shape[0], :min_shape[1], :min_shape[2]]
        root_mask = ((overall_mask > 0) & ~(lvot_mask > 0)).astype(np.uint8)
    else:
        logger.warning(f"  No lvot_mask.nii.gz, using overall_mask as root_mask")
        root_mask = (overall_mask > 0).astype(np.uint8)

    logger.info(
        f"  Root mask: {int(np.sum(root_mask))} vx "
        f"(overall {int(np.sum(overall_mask > 0))}, lvot {int(np.sum(overall_mask > 0)) - int(np.sum(root_mask))})"
    )

    if tissue_only:
        logger.info("  Tissue-only mode: skipping wall/valve/annular masks")
    else:
        # 2-3. Wall and valve masks (from root_mask, clipped to annular-STJ band)
        wall_mask, valve_mask = generate_wall_valve_masks(
            root_mask, cusp_masks, mask_spacing, metadata, origin_xyz,
            erosion_mm=EROSION_MM
        )

        # 4. Annular mask
        annular_mask = generate_annular_mask(
            overall_mask, metadata, spacing_zyx, origin_xyz,
            thickness_mm=ANNULAR_RING_THICKNESS_MM
        )

    # 5-8. Blood-pool-calibrated tissue classification
    # Calibration and classification both on root_mask (aortic root without LVOT)
    fibrotic_mask, calcific_mask, blood_pool_mask, tissue_params = classify_tissue_bp_calibrated(
        ct_cropped, root_mask,
        calibration_mask=root_mask,
        fibrotic_floor_hu=FIBROTIC_FLOOR_HU,
        median_kernel=BLOOD_POOL_MEDIAN_KERNEL,
    )

    # 7b. Calcific dilated = calcific threshold within overall_mask + 2mm dilation
    calcific_lower = tissue_params['thresholds']['calcific_lower_hu']

    if tissue_only:
        # Skip calcium (fixed HU>850) and calcific_dilated (needs EDT) -- unchanged
        calcium_mask = None
        calcium_mask_dilated = None
        calcific_mask_dilated = None
    else:
        # 7b. Calcific dilated = calcific threshold within overall_mask + 2mm dilation
        dist_outside = ndimage.distance_transform_edt(
            ~(overall_mask > 0), sampling=spacing_zyx
        )
        overall_dilated = (overall_mask > 0) | (dist_outside <= CALCIUM_PROXIMITY_MM)
        calcific_mask_dilated = (
            (ct_cropped.astype(np.float64) >= calcific_lower) & overall_dilated
        ).astype(np.uint8)

        # 9. Calcium (fixed HU>850) within root_mask
        calcium_mask = ((ct_cropped > CALCIUM_HU_THRESHOLD) & (root_mask > 0)).astype(np.uint8)
        logger.info(f"  Calcium (HU>{CALCIUM_HU_THRESHOLD}) in root: {int(np.sum(calcium_mask))} vx")

        # 10. Calcium dilated (fixed HU>850) within overall_mask + 2mm dilation
        calcium_mask_dilated = generate_calcium_mask(
            ct_cropped, overall_mask, spacing_zyx,
            hu_threshold=CALCIUM_HU_THRESHOLD,
            proximity_mm=CALCIUM_PROXIMITY_MM
        )

    # --- Save all outputs ---

    if tissue_only:
        masks_to_save = {
            'fibrotic_mask.nii.gz': fibrotic_mask,
            'calcific_mask.nii.gz': calcific_mask,
            'blood_pool_mask.nii.gz': blood_pool_mask,
        }
    else:
        masks_to_save = {
            'root_mask.nii.gz': root_mask,
            'wall_mask.nii.gz': wall_mask,
            'valve_mask.nii.gz': valve_mask,
            'annular_mask.nii.gz': annular_mask,
            'fibrotic_mask.nii.gz': fibrotic_mask,
            'calcific_mask.nii.gz': calcific_mask,
            'calcific_mask_dilated.nii.gz': calcific_mask_dilated,
            'blood_pool_mask.nii.gz': blood_pool_mask,
            'calcium_mask.nii.gz': calcium_mask,
            'calcium_mask_dilated.nii.gz': calcium_mask_dilated,
        }

    for filename, mask_data in masks_to_save.items():
        save_sitk_mask(mask_data, mask_spacing, mask_origin, seg_dir / filename)

    # Save tissue classification parameters
    tissue_params_path = seg_dir / 'tissue_params.json'
    with open(tissue_params_path, 'w') as f:
        json.dump(tissue_params, f, indent=2)

    saved_summary = ", ".join(
        f"{name.replace('.nii.gz', '')}={int(np.sum(data))}"
        for name, data in masks_to_save.items()
    )
    logger.info(f"  Saved: {saved_summary}")

    return True


# ===================================================================
# Check / validation
# ===================================================================

def check_case(case_name: str):
    """Validate all derived masks for a case."""

    seg_dir = SEGMENTATION_DIR / case_name

    mask_files = {
        'overall': 'overall_mask.nii.gz',
        'root': 'root_mask.nii.gz',
        'wall': 'wall_mask.nii.gz',
        'valve': 'valve_mask.nii.gz',
        'annular': 'annular_mask.nii.gz',
        'fibrotic': 'fibrotic_mask.nii.gz',
        'calcific': 'calcific_mask.nii.gz',
        'calcific_dil': 'calcific_mask_dilated.nii.gz',
        'blood_pool': 'blood_pool_mask.nii.gz',
        'calcium': 'calcium_mask.nii.gz',
        'calcium_dil': 'calcium_mask_dilated.nii.gz',
    }

    masks = {}
    spacing = None
    for name, filename in mask_files.items():
        path = seg_dir / filename
        if not path.exists():
            print(f"  Missing: {filename}")
            continue
        arr, sp, _ = load_sitk_mask(path)
        masks[name] = arr > 0
        if spacing is None:
            spacing = sp

    if 'overall' not in masks:
        print(f"  Cannot validate without overall_mask")
        return

    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    n_overall = int(np.sum(masks['overall']))
    n_root = int(np.sum(masks.get('root', np.zeros(1)))) if 'root' in masks else 0

    print(f"\n=== {case_name} ===")
    print(f"Spacing: {spacing[0]:.3f} x {spacing[1]:.3f} x {spacing[2]:.3f} mm")
    print(f"Voxel volume: {voxel_vol:.4f} mm3")

    print(f"\nMask volumes:")
    for name in mask_files:
        if name in masks:
            n = int(np.sum(masks[name]))
            vol = n * voxel_vol
            ref = n_root if name not in ('overall',) and n_root > 0 else n_overall
            pct = n / max(ref, 1)
            print(f"  {name:>12s}: {n:>8d} vx = {vol:>10.1f} mm3 ({pct:.1%} of {'root' if ref == n_root else 'overall'})")

    # Structural checks
    if 'wall' in masks and 'valve' in masks:
        overlap = int(np.sum(masks['wall'] & masks['valve']))
        print(f"\nWall-valve overlap: {overlap} (should be 0)")

    # Tissue classification checks
    if all(k in masks for k in ['fibrotic', 'blood_pool', 'calcific']):
        n_f = int(np.sum(masks['fibrotic']))
        n_bp = int(np.sum(masks['blood_pool']))
        n_c = int(np.sum(masks['calcific']))
        n_tissue_total = n_f + n_bp + n_c
        ref = n_root if n_root > 0 else n_overall
        print(f"\nTissue classification (within root):")
        print(f"  Fibrotic:    {n_f:>8d} vx ({n_f / max(ref, 1):.1%})")
        print(f"  Blood pool:  {n_bp:>8d} vx ({n_bp / max(ref, 1):.1%})")
        print(f"  Calcific:    {n_c:>8d} vx ({n_c / max(ref, 1):.1%})")
        print(f"  Classified:  {n_tissue_total:>8d} / {ref} ({n_tissue_total / max(ref, 1):.1%})")

    # Calcium comparison: fixed vs calibrated
    if 'calcium' in masks and 'calcific' in masks:
        ca_fixed = masks['calcium']
        ca_cal = masks['calcific']
        overlap = int(np.sum(ca_fixed & ca_cal))
        only_fixed = int(np.sum(ca_fixed & ~ca_cal))
        only_cal = int(np.sum(ca_cal & ~ca_fixed))
        print(f"\nCalcium fixed (HU>850) vs calibrated calcific (root only):")
        print(f"  Both:           {overlap:>8d} vx")
        print(f"  Fixed only:     {only_fixed:>8d} vx")
        print(f"  Calibrated only:{only_cal:>8d} vx")

    # Per-region tissue breakdown
    if 'wall' in masks and 'valve' in masks and 'fibrotic' in masks and 'calcific' in masks:
        print(f"\nPer-region tissue breakdown:")
        for region_name in ['wall', 'valve']:
            region = masks[region_name]
            f_in = int(np.sum(masks['fibrotic'] & region))
            c_in = int(np.sum(masks['calcific'] & region))
            bp_in = int(np.sum(masks.get('blood_pool', np.zeros_like(masks['overall'])) & region))
            n_region = int(np.sum(region))
            print(
                f"  {region_name:>6s}: fibrotic={f_in}, calcific={c_in}, "
                f"blood_pool={bp_in}, total={n_region}"
            )

    # Load tissue params if available
    tissue_file = seg_dir / 'tissue_params.json'
    if tissue_file.exists():
        with open(tissue_file) as f:
            params = json.load(f)
        bp = params.get('blood_pool', {})
        print(f"\nBlood pool calibration:")
        print(f"  Blood pool: mean={bp.get('mean_hu')} HU, SD={bp.get('sd_hu')} HU")
        thresholds = params.get('thresholds', {})
        print(f"  Fibrotic: [{thresholds.get('fibrotic_floor_hu')}, {thresholds.get('fibrotic_upper_hu')}) HU")
        print(f"  Calcific: [{thresholds.get('calcific_lower_hu')}, inf) HU")
        crosscheck = params.get('cartlidge_crosscheck', {})
        if crosscheck:
            print(f"  Cartlidge equation: {crosscheck.get('cartlidge_equation_upper')} HU")


# ===================================================================
# List cases
# ===================================================================

def list_cases():
    """List available cases and their derived mask status."""
    cases = sorted([d for d in SEGMENTATION_DIR.iterdir() if d.is_dir()])

    print(f"\nCases in {SEGMENTATION_DIR} ({len(cases)} total):\n")

    counts = {'complete': 0}

    for case_dir in cases:
        meta_file = case_dir / 'metadata.json'
        status = []

        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            if meta.get('ct_quality') == 'poor':
                status.append('poor')
            elif meta.get('annulus_set'):
                status.append('complete')
                counts['complete'] += 1
            else:
                status.append('incomplete')
        else:
            status.append('no_meta')

        derived = {
            'root': 'root_mask.nii.gz',
            'wall': 'wall_mask.nii.gz',
            'valve': 'valve_mask.nii.gz',
            'ann': 'annular_mask.nii.gz',
            'fib': 'fibrotic_mask.nii.gz',
            'calc': 'calcific_mask.nii.gz',
            'calc_d': 'calcific_mask_dilated.nii.gz',
            'Ca': 'calcium_mask.nii.gz',
            'Ca_d': 'calcium_mask_dilated.nii.gz',
            'tissue': 'tissue_params.json',
        }
        for short, filename in derived.items():
            if (case_dir / filename).exists():
                status.append(short)
                counts[short] = counts.get(short, 0) + 1

        print(f"  {case_dir.name}: {', '.join(status)}")

    print(f"\nSummary: {counts.get('complete', 0)} complete cases")
    for key in ['root', 'wall', 'valve', 'ann', 'fib', 'calc', 'Ca', 'tissue']:
        if key in counts:
            print(f"  {key}: {counts[key]}")


# ===================================================================
# CLI
# ===================================================================

def main():
    global SEGMENTATION_DIR, DICOM_DIR, TOTALSEG_DIR
    global CALCIUM_HU_THRESHOLD, EROSION_MM, FIBROTIC_FLOOR_HU

    parser = argparse.ArgumentParser(
        description='Generate all derived masks from aortic segmentation data'
    )
    parser.add_argument('--case_name', type=str, help='Process specific case')
    parser.add_argument('--all', action='store_true', help='Process all complete cases')
    parser.add_argument('--list', action='store_true', help='List cases and status')
    parser.add_argument(
        '--check', action='store_true',
        help='Validate masks for a case (use with --case_name)'
    )
    parser.add_argument('--seg_dir', type=str, default=str(SEGMENTATION_DIR))
    parser.add_argument('--dicom_dir', type=str, default=str(DICOM_DIR))
    parser.add_argument('--totalseg_dir', type=str, default=str(TOTALSEG_DIR))
    parser.add_argument(
        '--calcium_threshold', type=float, default=CALCIUM_HU_THRESHOLD,
        help=f'Fixed calcium HU threshold (default: {CALCIUM_HU_THRESHOLD})'
    )
    parser.add_argument(
        '--erosion_mm', type=float, default=EROSION_MM,
        help=f'Erosion depth for wall/valve split (default: {EROSION_MM})'
    )
    parser.add_argument(
        '--fibrotic_floor', type=float, default=FIBROTIC_FLOOR_HU,
        help=f'Lower HU floor for fibrotic tissue (default: {FIBROTIC_FLOOR_HU})'
    )
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing masks')
    parser.add_argument(
        '--tissue-only', action='store_true', dest='tissue_only',
        help='Recompute only tissue classification masks (fibrotic, calcific, calcium, blood_pool)'
    )
    parser.add_argument(
        '--wall-only', action='store_true', dest='wall_only',
        help='Recompute only wall_mask.nii.gz using current EROSION_MM (no DICOM loading)'
    )

    args = parser.parse_args()

    SEGMENTATION_DIR = Path(args.seg_dir)
    DICOM_DIR = Path(args.dicom_dir)
    TOTALSEG_DIR = Path(args.totalseg_dir)
    CALCIUM_HU_THRESHOLD = args.calcium_threshold
    EROSION_MM = args.erosion_mm
    FIBROTIC_FLOOR_HU = args.fibrotic_floor

    if args.list:
        list_cases()
        return

    if args.check:
        if not args.case_name:
            print("--check requires --case_name")
            return
        check_case(args.case_name)
        return

    if args.case_name:
        process_case(args.case_name, overwrite=True, tissue_only=args.tissue_only,
                     wall_only=args.wall_only)
        return

    if args.all:
        cases = sorted([d.name for d in SEGMENTATION_DIR.iterdir() if d.is_dir()])

        success = 0
        skipped = 0
        failed = 0

        for case_name in cases:
            try:
                result = process_case(case_name, overwrite=args.overwrite,
                                      tissue_only=args.tissue_only,
                                      wall_only=args.wall_only)
                if result:
                    success += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.error(f"Failed {case_name}: {e}")
                failed += 1

        logger.info(
            f"\nComplete: {success} processed, {skipped} skipped, {failed} failed"
        )
        return

    parser.print_help()


if __name__ == '__main__':
    main()