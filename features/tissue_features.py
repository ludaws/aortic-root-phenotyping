#!/usr/bin/env python3
"""
Tissue Feature Extraction: Polar Bullseye and Cylindrical Unwrap Projections.

For each annotated case, computes:
  - Polar bullseye projection (eroded root interior, wall excluded) of
    calcific and fibrotic tissue, mapped to a 20 x 120 radial-angular grid.
  - Cylindrical unwrap projection (overall root + LVOT, 2 mm-dilated for
    wall-adherent calcium) of calcific and fibrotic tissue, mapped to a
    60 x 120 longitudinal-angular grid.
  - 18 polar zonal fractions per tissue (3 cusps x 3 radial rings, with
    middle and outer rings split at the cusp midline).
  - 18 unwrap zonal fractions per tissue (3 height bands x 3 cusps x 2 halves).
  - Total and per-cusp tissue volumes (mm^3).

Outputs are written to a per-case .npz cache for downstream clustering and
plotting, plus a JSON summary of volumes and zonal fractions per case.

Usage:
    python tissue_features.py --case <case_name>      # single case
    python tissue_features.py --all                   # batch over all cases
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SEGMENTATION_DIR = Path("data/segmentations")
TOTALSEG_DIR = Path("data/totalseg_outputs")
OUTPUT_DIR = Path("results/tissue_features")
CACHE_DIR = Path("results/tissue_features/.cache")

# Polar grid resolution
N_RADIAL_BINS = 20
N_ANGULAR_BINS = 120
RADIAL_EXTENT = 1.3

# Unwrap grid resolution
N_LONGITUDINAL_BINS = 60
N_UNWRAP_ANGULAR_BINS = 120

# Polar zone definitions: 3 radial rings as fraction of annular radius
RING_EDGES = [0.0, 0.40, 0.70, 1.0]  # inner, middle, outer
RING_NAMES = ['inner', 'middle', 'outer']

# Canonical cusp angles for display (NCC at top)
CANONICAL_CUSP_ANGLES_DEG = {
    'right': 330, 'left': 210, 'non': 90,
}

CUSP_COLORS = {
    'right': '#4393C3', 'left': '#D6604D', 'non': '#92C5DE',
    'anterior': '#D6604D', 'posterior': '#4393C3',
}


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def load_sitk_mask(path: Path) -> Tuple[np.ndarray, list, list]:
    """Load NIfTI mask. Returns (array_zyx, spacing_xyz, origin_xyz)."""
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img).astype(np.float32), list(img.GetSpacing()), list(img.GetOrigin())


def load_case_data(case_name: str, seg_dir: Path, totalseg_dir: Path) -> Optional[Dict]:
    """Load all masks and metadata needed for tissue feature extraction."""
    case_dir = seg_dir / case_name
    metadata_file = case_dir / 'metadata.json'

    if not metadata_file.exists():
        logger.warning(f"No metadata.json for {case_name}")
        return None

    with open(metadata_file) as f:
        metadata = json.load(f)

    if metadata.get('ct_quality') == 'poor':
        return None
    if not metadata.get('annulus_set'):
        return None
    if not metadata.get('commissure_points') or not metadata.get('central_coaptation_point'):
        logger.warning(f"No commissure/CC points for {case_name}")
        return None

    # Required masks
    required_masks = {
        'root_mask': 'root_mask.nii.gz',
        'overall_mask': 'overall_mask.nii.gz',
        'calcific_mask': 'calcific_mask.nii.gz',
        'fibrotic_mask': 'fibrotic_mask.nii.gz',
        'blood_pool_mask': 'blood_pool_mask.nii.gz',
        'calcific_mask_dilated': 'calcific_mask_dilated.nii.gz',
        'wall_mask': 'wall_mask.nii.gz',
    }

    for key, fname in required_masks.items():
        if not (case_dir / fname).exists():
            logger.warning(f"Missing {fname} for {case_name}")
            return None

    masks = {}
    spacing_xyz = None
    origin_xyz = None
    for key, fname in required_masks.items():
        arr, sp, orig = load_sitk_mask(case_dir / fname)
        masks[key] = arr
        if spacing_xyz is None:
            spacing_xyz = sp
            origin_xyz = orig

    # Coordinate system from crop_info
    crop_info_file = totalseg_dir / case_name / 'crop_info.json'
    if crop_info_file.exists():
        with open(crop_info_file) as f:
            crop_info = json.load(f)
        spacing_xyz = crop_info['spacing']
        origin_xyz = crop_info['cropped_origin']

    annulus_plane = metadata['annulus_plane']
    annulus_center = np.array(annulus_plane['center'])
    annulus_normal = np.array(annulus_plane['normal'])
    annulus_normal = annulus_normal / np.linalg.norm(annulus_normal)

    stj_center = np.array(metadata['stj_plane']['center'])

    commissure_points = [np.array(p) for p in metadata['commissure_points']]
    cc_point = np.array(metadata['central_coaptation_point'])

    valve_type = metadata.get('valve_type', 'tricuspid')
    cusp_labels = metadata.get('cusp_labels', ['right', 'left', 'non'])
    if valve_type == 'type0' and len(commissure_points) == 2:
        cusp_labels = ['anterior', 'posterior']

    # Load optional cusp masks for per-cusp volume computation
    if valve_type == 'type0':
        cusp_mask_files = {f'cusp_{l}_mask': f'cusp_{l}_mask.nii.gz' for l in ['A', 'P']}
    else:
        cusp_mask_files = {f'cusp_{l}_mask': f'cusp_{l}_mask.nii.gz' for l in ['L', 'R', 'N']}
    for key, fname in cusp_mask_files.items():
        cusp_path = case_dir / fname
        if cusp_path.exists():
            arr, _, _ = load_sitk_mask(cusp_path)
            masks[key] = arr

    # Commissural plane center (average of commissure tips)
    comm_center = np.mean(np.array(commissure_points), axis=0)
    # Project onto annulus normal to get height
    comm_t = np.dot(comm_center - annulus_center, annulus_normal)

    # LVOT plane: 4mm below annulus
    lvot_center = annulus_center - 4.0 * annulus_normal

    return {
        'case_name': case_name,
        'metadata': metadata,
        'spacing_xyz': spacing_xyz,
        'origin_xyz': origin_xyz,
        'annulus_center': annulus_center,
        'annulus_normal': annulus_normal,
        'stj_center': stj_center,
        'lvot_center': lvot_center,
        'comm_center_t': comm_t,  # signed distance along normal from annulus
        'commissure_points': commissure_points,
        'cc_point': cc_point,
        'valve_type': valve_type,
        'cusp_labels': cusp_labels,
        **masks,
    }


# ------------------------------------------------------------------
# Geometry utilities
# ------------------------------------------------------------------
def build_plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build orthonormal basis in the plane perpendicular to normal."""
    if abs(normal[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    ref_vec = ref - np.dot(ref, normal) * normal
    ref_vec = ref_vec / np.linalg.norm(ref_vec)
    perp_vec = np.cross(normal, ref_vec)
    perp_vec = perp_vec / np.linalg.norm(perp_vec)
    return ref_vec, perp_vec


def compute_commissure_angles(
    commissure_points: List[np.ndarray],
    cc_point: np.ndarray,
    annulus_normal: np.ndarray,
    ref_vec: np.ndarray,
    perp_vec: np.ndarray,
) -> np.ndarray:
    """Compute angles (radians) of commissure points relative to CC in annulus plane."""
    angles = []
    for cp in commissure_points:
        diff = cp - cc_point
        diff_in_plane = diff - np.dot(diff, annulus_normal) * annulus_normal
        x = np.dot(diff_in_plane, ref_vec)
        y = np.dot(diff_in_plane, perp_vec)
        angles.append(np.arctan2(y, x))
    return np.array(angles)


def compute_rotation_offset(
    commissure_angles: np.ndarray,
    cusp_labels: List[str],
    valve_type: str,
) -> float:
    """Compute angular offset to rotate into canonical orientation (NCC at top)."""
    if valve_type == 'type0':
        midpoint = (commissure_angles[0] + commissure_angles[1]) / 2.0
        if abs(commissure_angles[1] - commissure_angles[0]) > np.pi:
            midpoint += np.pi
        return np.radians(90) - midpoint
    else:
        a_nl = commissure_angles[2]
        a_rn = commissure_angles[0]
        span = (a_rn - a_nl) % (2 * np.pi)
        ncc_mid = a_nl + span / 2.0
        return np.radians(CANONICAL_CUSP_ANGLES_DEG['non']) - ncc_mid


def compute_cusp_midline_angles(
    commissure_angles_canonical: np.ndarray,
    cusp_labels: List[str],
) -> np.ndarray:
    """
    Compute the midline angle for each cusp (midpoint between bounding commissures).
    Returns array of angles in radians, one per cusp.
    """
    n_comm = len(commissure_angles_canonical)
    midlines = np.zeros(n_comm)
    for i in range(n_comm):
        start = commissure_angles_canonical[i]
        end = commissure_angles_canonical[(i + 1) % n_comm]
        span = (end - start) % (2 * np.pi)
        midlines[i] = (start + span / 2.0) % (2 * np.pi)
    return midlines


def compute_annular_radius(
    mask: np.ndarray,
    spacing_xyz: list,
    origin_xyz: list,
    annulus_center: np.ndarray,
    annulus_normal: np.ndarray,
    ref_vec: np.ndarray,
    perp_vec: np.ndarray,
    slab_thickness_mm: float = 3.0,
) -> float:
    """Estimate annular radius from mask voxels near the annular plane."""
    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz

    zz, yy, xx = np.where(mask > 0)
    if len(zz) == 0:
        return 15.0

    phys_x = ox + xx * sx
    phys_y = oy + yy * sy
    phys_z = oz + zz * sz

    dx = phys_x - annulus_center[0]
    dy = phys_y - annulus_center[1]
    dz = phys_z - annulus_center[2]
    dist_to_plane = np.abs(
        dx * annulus_normal[0] + dy * annulus_normal[1] + dz * annulus_normal[2]
    )

    near_plane = dist_to_plane < slab_thickness_mm
    if np.sum(near_plane) == 0:
        return 15.0

    proj_ref = dx[near_plane] * ref_vec[0] + dy[near_plane] * ref_vec[1] + dz[near_plane] * ref_vec[2]
    proj_perp = dx[near_plane] * perp_vec[0] + dy[near_plane] * perp_vec[1] + dz[near_plane] * perp_vec[2]
    in_plane_dist = np.sqrt(proj_ref**2 + proj_perp**2)

    return float(np.percentile(in_plane_dist, 95))


# ------------------------------------------------------------------
# Polar projection engine
# ------------------------------------------------------------------
def project_mask_to_polar(
    mask: np.ndarray,
    spacing_xyz: list,
    origin_xyz: list,
    annulus_center: np.ndarray,
    annulus_normal: np.ndarray,
    ref_vec: np.ndarray,
    perp_vec: np.ndarray,
    annular_radius: float,
    rotation_offset: float,
    n_radial: int = N_RADIAL_BINS,
    n_angular: int = N_ANGULAR_BINS,
    radial_extent: float = RADIAL_EXTENT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D binary mask onto 2D polar grid (r, theta).
    Returns (polar_grid, count_grid) with volumes in mm^3.
    """
    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz

    zz, yy, xx = np.where(mask > 0)
    if len(zz) == 0:
        return np.zeros((n_radial, n_angular)), np.zeros((n_radial, n_angular))

    phys_x = ox + xx.astype(np.float64) * sx
    phys_y = oy + yy.astype(np.float64) * sy
    phys_z = oz + zz.astype(np.float64) * sz

    dx = phys_x - annulus_center[0]
    dy = phys_y - annulus_center[1]
    dz = phys_z - annulus_center[2]

    proj_ref = dx * ref_vec[0] + dy * ref_vec[1] + dz * ref_vec[2]
    proj_perp = dx * perp_vec[0] + dy * perp_vec[1] + dz * perp_vec[2]

    r = np.sqrt(proj_ref**2 + proj_perp**2) / annular_radius
    theta = (np.arctan2(proj_perp, proj_ref) + rotation_offset) % (2 * np.pi)

    r_bins = np.linspace(0, radial_extent, n_radial + 1)
    theta_bins = np.linspace(0, 2 * np.pi, n_angular + 1)

    r_idx = np.clip(np.digitize(r, r_bins) - 1, 0, n_radial - 1)
    theta_idx = np.clip(np.digitize(theta, theta_bins) - 1, 0, n_angular - 1)

    voxel_vol = sx * sy * sz
    polar_grid = np.zeros((n_radial, n_angular))
    count_grid = np.zeros((n_radial, n_angular))

    np.add.at(polar_grid, (r_idx, theta_idx), voxel_vol)
    np.add.at(count_grid, (r_idx, theta_idx), 1)

    return polar_grid, count_grid


# ------------------------------------------------------------------
# Unwrap projection engine
# ------------------------------------------------------------------
def unwrap_to_cylindrical(
    mask: np.ndarray,
    spacing_xyz: list,
    origin_xyz: list,
    annulus_center: np.ndarray,
    annulus_normal: np.ndarray,
    lvot_center: np.ndarray,
    stj_center: np.ndarray,
    ref_vec: np.ndarray,
    perp_vec: np.ndarray,
    rotation_offset: float,
    n_angular: int = N_UNWRAP_ANGULAR_BINS,
    n_longitudinal: int = N_LONGITUDINAL_BINS,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Unwrap 3D mask onto cylindrical (longitudinal, angular) grid.
    Returns (unwrap_grid, count_grid, annulus_frac).
    """
    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz

    dist_lvot = np.dot(lvot_center - annulus_center, annulus_normal)
    dist_stj = np.dot(stj_center - annulus_center, annulus_normal)
    t_min = min(dist_lvot, dist_stj)
    t_max = max(dist_lvot, dist_stj)

    if abs(t_max - t_min) < 1.0:
        return np.zeros((n_longitudinal, n_angular)), np.zeros((n_longitudinal, n_angular)), 0.5

    annulus_frac = (0.0 - t_min) / (t_max - t_min)

    zz, yy, xx = np.where(mask > 0)
    if len(zz) == 0:
        return np.zeros((n_longitudinal, n_angular)), np.zeros((n_longitudinal, n_angular)), annulus_frac

    phys_x = ox + xx.astype(np.float64) * sx
    phys_y = oy + yy.astype(np.float64) * sy
    phys_z = oz + zz.astype(np.float64) * sz

    dx = phys_x - annulus_center[0]
    dy = phys_y - annulus_center[1]
    dz = phys_z - annulus_center[2]

    t_phys = dx * annulus_normal[0] + dy * annulus_normal[1] + dz * annulus_normal[2]
    t_norm = (t_phys - t_min) / (t_max - t_min)

    proj_ref = dx * ref_vec[0] + dy * ref_vec[1] + dz * ref_vec[2]
    proj_perp = dx * perp_vec[0] + dy * perp_vec[1] + dz * perp_vec[2]
    theta = (np.arctan2(proj_perp, proj_ref) + rotation_offset) % (2 * np.pi)

    t_bins = np.linspace(0, 1, n_longitudinal + 1)
    theta_bins = np.linspace(0, 2 * np.pi, n_angular + 1)

    t_idx = np.clip(np.digitize(t_norm, t_bins) - 1, 0, n_longitudinal - 1)
    theta_idx = np.clip(np.digitize(theta, theta_bins) - 1, 0, n_angular - 1)

    voxel_vol = sx * sy * sz
    unwrap_grid = np.zeros((n_longitudinal, n_angular))
    count_grid = np.zeros((n_longitudinal, n_angular))

    np.add.at(unwrap_grid, (t_idx, theta_idx), voxel_vol)
    np.add.at(count_grid, (t_idx, theta_idx), 1)

    return unwrap_grid, count_grid, annulus_frac


# ------------------------------------------------------------------
# Zone assignment
# ------------------------------------------------------------------
def assign_polar_zones(
    commissure_angles_canonical: np.ndarray,
    cusp_labels: List[str],
    n_radial: int = N_RADIAL_BINS,
    n_angular: int = N_ANGULAR_BINS,
    radial_extent: float = RADIAL_EXTENT,
) -> np.ndarray:
    """
    Assign each (r, theta) bin to one of 18 polar zones (Abdelkhalek scheme).

    Per cusp (6 zones each):
        Inner ring:  1 zone  (no circumferential split)
        Middle ring: 2 zones (split at cusp midline)
        Outer ring:  3 zones (split into thirds)
    Total: 3 cusps x 6 = 18

    Returns: (n_radial, n_angular) array of zone indices 0-17.
             -1 for bins outside the annular boundary (r > 1.0).

    Zone numbering per cusp (cusp_idx * 6 + offset):
        0: inner (whole cusp)
        1: middle_a (comm side of midline)
        2: middle_b (other side of midline)
        3: outer_a (comm side third)
        4: outer_b (middle third)
        5: outer_c (next-comm side third)
    """
    n_comm = len(commissure_angles_canonical)
    midlines = compute_cusp_midline_angles(commissure_angles_canonical, cusp_labels)

    r_edges = np.linspace(0, radial_extent, n_radial + 1)
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2.0
    theta_centers = np.linspace(0, 2 * np.pi, n_angular, endpoint=False) + np.pi / n_angular

    zone_map = np.full((n_radial, n_angular), -1, dtype=np.int32)

    for ri in range(n_radial):
        r = r_centers[ri]
        if r > 1.0:
            continue

        # Ring index: 0=inner, 1=middle, 2=outer
        ring_idx = -1
        for k in range(len(RING_EDGES) - 1):
            if RING_EDGES[k] <= r < RING_EDGES[k + 1]:
                ring_idx = k
                break
        if ring_idx == -1:
            if r >= RING_EDGES[-1]:
                ring_idx = len(RING_EDGES) - 2
            else:
                continue

        for ai in range(n_angular):
            th = theta_centers[ai]

            for ci in range(n_comm):
                start = commissure_angles_canonical[ci]
                end = commissure_angles_canonical[(ci + 1) % n_comm]
                span = (end - start) % (2 * np.pi)
                angle_from_start = (th - start) % (2 * np.pi)

                if angle_from_start <= span:
                    base = ci * 6
                    frac_in_cusp = angle_from_start / span  # 0..1 within cusp

                    if ring_idx == 0:
                        # Inner: 1 zone per cusp
                        zone_map[ri, ai] = base + 0
                    elif ring_idx == 1:
                        # Middle: 2 zones split at midline
                        mid = midlines[ci]
                        mid_from_start = (mid - start) % (2 * np.pi)
                        if angle_from_start <= mid_from_start:
                            zone_map[ri, ai] = base + 1
                        else:
                            zone_map[ri, ai] = base + 2
                    else:
                        # Outer: 3 zones in thirds
                        if frac_in_cusp < 1.0 / 3.0:
                            zone_map[ri, ai] = base + 3
                        elif frac_in_cusp < 2.0 / 3.0:
                            zone_map[ri, ai] = base + 4
                        else:
                            zone_map[ri, ai] = base + 5
                    break

    return zone_map


def assign_unwrap_zones(
    commissure_angles_canonical: np.ndarray,
    cusp_labels: List[str],
    annulus_frac: float,
    comm_frac: float,
    n_longitudinal: int = N_LONGITUDINAL_BINS,
    n_angular: int = N_UNWRAP_ANGULAR_BINS,
) -> np.ndarray:
    """
    Assign each (t, theta) bin to one of 18 unwrap zones.

    Zones: 3 height bands x 3 cusps x 2 halves
    Height bands: LVOT-to-annulus, annulus-to-commissural, commissural-to-STJ
    Returns: (n_longitudinal, n_angular) array of zone indices 0-17.

    Zone numbering:
        band_idx * 6 + cusp_idx * 2 + half_idx
        where band_idx: 0=LVOT-ann, 1=ann-comm, 2=comm-STJ
              cusp_idx: 0=first, 1=second, 2=third
              half_idx: 0=first_half, 1=second_half
    """
    n_comm = len(commissure_angles_canonical)
    midlines = compute_cusp_midline_angles(commissure_angles_canonical, cusp_labels)

    t_centers = np.linspace(0, 1, n_longitudinal, endpoint=False) + 0.5 / n_longitudinal
    theta_centers = np.linspace(0, 2 * np.pi, n_angular, endpoint=False) + np.pi / n_angular

    # Height band edges (in normalized t space: 0=LVOT, 1=STJ)
    band_edges = [0.0, annulus_frac, comm_frac, 1.0]

    zone_map = np.full((n_longitudinal, n_angular), -1, dtype=np.int32)

    for ti in range(n_longitudinal):
        t = t_centers[ti]

        # Band index
        band_idx = -1
        for b in range(3):
            if band_edges[b] <= t < band_edges[b + 1]:
                band_idx = b
                break
        if band_idx == -1:
            if t >= band_edges[-1]:
                band_idx = 2
            else:
                continue

        for ai in range(n_angular):
            th = theta_centers[ai]

            for ci in range(n_comm):
                start = commissure_angles_canonical[ci]
                end = commissure_angles_canonical[(ci + 1) % n_comm]
                span = (end - start) % (2 * np.pi)
                angle_from_start = (th - start) % (2 * np.pi)

                if angle_from_start <= span:
                    mid = midlines[ci]
                    mid_from_start = (mid - start) % (2 * np.pi)
                    half_idx = 0 if angle_from_start <= mid_from_start else 1
                    zone_map[ti, ai] = band_idx * 6 + ci * 2 + half_idx
                    break

    return zone_map


def extract_zonal_fractions(
    tissue_grid: np.ndarray,
    volume_grid: np.ndarray,
    zone_map: np.ndarray,
    n_zones: int = 18,
) -> np.ndarray:
    """
    Compute tissue fraction per zone: sum(tissue_grid in zone) / sum(volume_grid in zone).
    Returns array of length n_zones.
    """
    fractions = np.zeros(n_zones)
    for z in range(n_zones):
        mask = zone_map == z
        zone_vol = np.sum(volume_grid[mask])
        if zone_vol > 0:
            fractions[z] = np.sum(tissue_grid[mask]) / zone_vol
    return fractions


# ------------------------------------------------------------------
# Zone naming
# ------------------------------------------------------------------
def polar_zone_names(cusp_labels: List[str]) -> List[str]:
    """Generate ordered names for 18 polar zones (Abdelkhalek scheme).
    Per cusp: inner(1), middle_a, middle_b(2), outer_a, outer_b, outer_c(3) = 6."""
    names = []
    for cusp in cusp_labels:
        names.append(f"{cusp}_inner")
        names.append(f"{cusp}_middle_a")
        names.append(f"{cusp}_middle_b")
        names.append(f"{cusp}_outer_a")
        names.append(f"{cusp}_outer_b")
        names.append(f"{cusp}_outer_c")
    return names


def unwrap_zone_names(cusp_labels: List[str]) -> List[str]:
    """Generate ordered names for 18 unwrap zones."""
    band_names = ['lvot_ann', 'ann_comm', 'comm_stj']
    half_names = ['a', 'b']
    names = []
    for bi, band in enumerate(band_names):
        for ci, cusp in enumerate(cusp_labels):
            for hi in range(2):
                names.append(f"{band}_{cusp}_{half_names[hi]}")
    return names


# ------------------------------------------------------------------
# Main processing
# ------------------------------------------------------------------
def process_single_case(case_data: Dict) -> Optional[Dict]:
    """
    Process one case: compute polar + unwrap projections and extract zone features.
    """
    ref_vec, perp_vec = build_plane_basis(case_data['annulus_normal'])

    comm_angles = compute_commissure_angles(
        case_data['commissure_points'],
        case_data['cc_point'],
        case_data['annulus_normal'],
        ref_vec, perp_vec,
    )

    rotation_offset = compute_rotation_offset(
        comm_angles, case_data['cusp_labels'], case_data['valve_type']
    )

    comm_canonical = (comm_angles + rotation_offset) % (2 * np.pi)

    annular_radius = compute_annular_radius(
        case_data['root_mask'],
        case_data['spacing_xyz'], case_data['origin_xyz'],
        case_data['annulus_center'], case_data['annulus_normal'],
        ref_vec, perp_vec,
    )

    if annular_radius < 5.0:
        logger.warning(f"Annular radius too small ({annular_radius:.1f}mm) for {case_data['case_name']}")
        return None

    # ---- POLAR PROJECTIONS (eroded root interior — wall_mask excluded) ----
    # Eroded interior: root_mask minus the 2mm wall shell.
    # Restricts polar bullseye to leaflet tissue, excluding wall calcium and
    # fibrosis projected from the sinus/STJ region.
    # wall_mask was generated with plane-aware EDT so annular and STJ faces
    # are not treated as erosion surfaces. Unwrap is unchanged.
    eroded_interior = (case_data['root_mask'] > 0) & (case_data['wall_mask'] == 0)

    # Project eroded root volume (denominator for fractions)
    polar_root, root_counts = project_mask_to_polar(
        eroded_interior.astype(np.float32),
        case_data['spacing_xyz'], case_data['origin_xyz'],
        case_data['annulus_center'], case_data['annulus_normal'],
        ref_vec, perp_vec, annular_radius, rotation_offset,
    )

    # Project calcific tissue within eroded interior
    polar_calcific, _ = project_mask_to_polar(
        ((case_data['calcific_mask'] > 0) & eroded_interior).astype(np.float32),
        case_data['spacing_xyz'], case_data['origin_xyz'],
        case_data['annulus_center'], case_data['annulus_normal'],
        ref_vec, perp_vec, annular_radius, rotation_offset,
    )

    # Project fibrotic tissue within eroded interior
    polar_fibrotic, _ = project_mask_to_polar(
        ((case_data['fibrotic_mask'] > 0) & eroded_interior).astype(np.float32),
        case_data['spacing_xyz'], case_data['origin_xyz'],
        case_data['annulus_center'], case_data['annulus_normal'],
        ref_vec, perp_vec, annular_radius, rotation_offset,
    )

    # Total volumes (raw mm^3, indexing to annular area done in geometric feature extraction)
    voxel_vol = case_data['spacing_xyz'][0] * case_data['spacing_xyz'][1] * case_data['spacing_xyz'][2]
    total_calcific_vol = float(np.sum(case_data['calcific_mask'] > 0)) * voxel_vol
    total_fibrotic_vol = float(np.sum(case_data['fibrotic_mask'] > 0)) * voxel_vol

    # Per-cusp volumes (within eroded interior, intersected with cusp masks)
    cusp_calcific_vols = {}
    cusp_fibrotic_vols = {}
    cusp_labels = case_data['cusp_labels']
    valve_type = case_data['valve_type']
    if valve_type == 'type0':
        cusp_mask_keys = {'anterior': 'cusp_A_mask', 'posterior': 'cusp_P_mask'}
    else:
        cusp_mask_keys = {'right': 'cusp_R_mask', 'left': 'cusp_L_mask', 'non': 'cusp_N_mask'}
    for cusp_name, mask_key in cusp_mask_keys.items():
        if mask_key in case_data:
            cusp_region = (case_data[mask_key] > 0) & eroded_interior
            cusp_calcific_vols[cusp_name] = float(np.sum((case_data['calcific_mask'] > 0) & cusp_region)) * voxel_vol
            cusp_fibrotic_vols[cusp_name] = float(np.sum((case_data['fibrotic_mask'] > 0) & cusp_region)) * voxel_vol
        else:
            cusp_calcific_vols[cusp_name] = 0.0
            cusp_fibrotic_vols[cusp_name] = 0.0

    # ---- UNWRAP PROJECTION (calcific_mask_dilated, full overall_mask extent) ----
    # Project dilated calcific mask
    unwrap_calcific, ca_counts, annulus_frac = unwrap_to_cylindrical(
        (case_data['calcific_mask_dilated'] > 0).astype(np.float32),
        case_data['spacing_xyz'], case_data['origin_xyz'],
        case_data['annulus_center'], case_data['annulus_normal'],
        case_data['lvot_center'], case_data['stj_center'],
        ref_vec, perp_vec, rotation_offset,
    )

    # Project overall_mask + dilation volume (denominator)
    # Use overall_mask as denominator since calcific_mask_dilated is within dilated overall
    unwrap_volume, vol_counts, _ = unwrap_to_cylindrical(
        (case_data['overall_mask'] > 0).astype(np.float32),
        case_data['spacing_xyz'], case_data['origin_xyz'],
        case_data['annulus_center'], case_data['annulus_normal'],
        case_data['lvot_center'], case_data['stj_center'],
        ref_vec, perp_vec, rotation_offset,
    )

    # Project fibrotic mask (no dilation — fibrosis is diffuse, dilation not appropriate)
    # Uses same overall_mask extent (LVOT to STJ) as calcific unwrap
    unwrap_fibrotic, _, _ = unwrap_to_cylindrical(
        (case_data['fibrotic_mask'] > 0).astype(np.float32),
        case_data['spacing_xyz'], case_data['origin_xyz'],
        case_data['annulus_center'], case_data['annulus_normal'],
        case_data['lvot_center'], case_data['stj_center'],
        ref_vec, perp_vec, rotation_offset,
    )

    # Commissural plane fraction along longitudinal axis
    dist_lvot = np.dot(case_data['lvot_center'] - case_data['annulus_center'], case_data['annulus_normal'])
    dist_stj = np.dot(case_data['stj_center'] - case_data['annulus_center'], case_data['annulus_normal'])
    t_min = min(dist_lvot, dist_stj)
    t_max = max(dist_lvot, dist_stj)
    comm_frac = (case_data['comm_center_t'] - t_min) / (t_max - t_min) if abs(t_max - t_min) > 1.0 else 0.5

    return {
        'case_name': case_data['case_name'],
        'valve_type': case_data['valve_type'],
        'cusp_labels': case_data['cusp_labels'],
        'annular_radius': annular_radius,
        'commissure_angles_canonical': comm_canonical,
        'annulus_frac': annulus_frac,
        'comm_frac': comm_frac,
        # Polar grids (for plotting)
        'polar_calcific': polar_calcific,
        'polar_fibrotic': polar_fibrotic,
        'polar_root': polar_root,
        # Unwrap grids (for plotting)
        'unwrap_calcific': unwrap_calcific,
        'unwrap_fibrotic': unwrap_fibrotic,
        'unwrap_volume': unwrap_volume,
        # Per-cusp volumes (raw mm^3)
        'cusp_calcific_vols': cusp_calcific_vols,
        'cusp_fibrotic_vols': cusp_fibrotic_vols,
        # Total volumes (raw, indexing done separately)
        'total_calcific_vol_mm3': total_calcific_vol,
        'total_fibrotic_vol_mm3': total_fibrotic_vol,
    }


# ------------------------------------------------------------------
# Feature output
# ------------------------------------------------------------------
def save_features_json(result: Dict, output_dir: Path):
    """Save tissue features as JSON for one case."""
    output_dir.mkdir(parents=True, exist_ok=True)

    features = {
        'case_name': result['case_name'],
        'valve_type': result['valve_type'],
        'cusp_labels': result['cusp_labels'],
        'total_calcific_vol_mm3': result['total_calcific_vol_mm3'],
        'total_fibrotic_vol_mm3': result['total_fibrotic_vol_mm3'],
        'cusp_calcific_vols_mm3': result['cusp_calcific_vols'],
        'cusp_fibrotic_vols_mm3': result['cusp_fibrotic_vols'],
    }

    path = output_dir / f"{result['case_name']}_tissue_features.json"
    with open(path, 'w') as f:
        json.dump(features, f, indent=2)


# ------------------------------------------------------------------
# Cache I/O (for plotting)
# ------------------------------------------------------------------
def save_result_cache(result: Dict, cache_dir: Path):
    """Save polar/unwrap grids to npz for population plotting."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{result['case_name']}.npz"

    np.savez_compressed(
        str(path),
        polar_calcific=result['polar_calcific'],
        polar_fibrotic=result['polar_fibrotic'],
        polar_root=result['polar_root'],
        unwrap_calcific=result['unwrap_calcific'],
        unwrap_fibrotic=result['unwrap_fibrotic'],
        unwrap_volume=result['unwrap_volume'],
        commissure_angles_canonical=result['commissure_angles_canonical'],
        annular_radius=np.array([result['annular_radius']]),
        annulus_frac=np.array([result['annulus_frac']]),
        comm_frac=np.array([result['comm_frac']]),
        valve_type=np.array([result['valve_type']]),
        cusp_labels=np.array(result['cusp_labels']),
        case_name=np.array([result['case_name']]),
    )


def load_result_cache(path: Path) -> Dict:
    d = np.load(str(path), allow_pickle=False)
    return {
        'polar_calcific': d['polar_calcific'],
        'polar_fibrotic': d['polar_fibrotic'],
        'polar_root': d['polar_root'],
        'unwrap_calcific': d['unwrap_calcific'],
        'unwrap_fibrotic': d['unwrap_fibrotic'],
        'unwrap_volume': d['unwrap_volume'],
        'commissure_angles_canonical': d['commissure_angles_canonical'],
        'annular_radius': float(d['annular_radius'][0]),
        'annulus_frac': float(d['annulus_frac'][0]),
        'comm_frac': float(d['comm_frac'][0]),
        'valve_type': str(d['valve_type'][0]),
        'cusp_labels': list(d['cusp_labels']),
        'case_name': str(d['case_name'][0]),
    }


def load_all_cached(cache_dir: Path) -> List[Dict]:
    results = []
    for f in sorted(cache_dir.glob('*.npz')):
        try:
            results.append(load_result_cache(f))
        except Exception as e:
            logger.warning(f"Failed to load {f.name}: {e}")
    logger.info(f"Loaded {len(results)} cached results from {cache_dir}")
    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def run_compute(
    seg_dir: Path, totalseg_dir: Path,
    cache_dir: Path, feature_dir: Path,
    force: bool = False,
):
    """Compute all projections, extract features, save to cache + JSON."""
    case_names = sorted([d.name for d in seg_dir.iterdir() if d.is_dir()])
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    computed = 0
    skipped = 0
    failed = 0

    for case_name in case_names:
        if not force and (cache_dir / f"{case_name}.npz").exists():
            skipped += 1
            continue

        case_data = load_case_data(case_name, seg_dir, totalseg_dir)
        if case_data is None:
            continue

        result = process_single_case(case_data)
        if result is not None:
            save_result_cache(result, cache_dir)
            save_features_json(result, feature_dir)
            computed += 1
            logger.info(
                f"  {case_name}: calcific={result['total_calcific_vol_mm3']:.1f}mm3, "
                f"fibrotic={result['total_fibrotic_vol_mm3']:.1f}mm3"
            )
        else:
            failed += 1

    logger.info(f"\nCompute complete: {computed} new, {skipped} cached, {failed} failed")


def run_single_case(
    case_name: str, seg_dir: Path, totalseg_dir: Path,
    output_dir: Path,
):
    """Process single case: compute features and print summary."""
    case_data = load_case_data(case_name, seg_dir, totalseg_dir)
    if case_data is None:
        logger.error(f"Could not load {case_name}")
        return

    result = process_single_case(case_data)
    if result is None:
        logger.error(f"Processing failed for {case_name}")
        return

    save_features_json(result, output_dir / 'features')
    save_result_cache(result, output_dir / '.cache')

    # Print summary
    cusp_labels = result['cusp_labels']
    p_names = polar_zone_names(cusp_labels)
    u_names = unwrap_zone_names(cusp_labels)

    print(f"\n{'='*60}")
    print(f"Tissue Features: {case_name}")
    print(f"{'='*60}")
    print(f"Valve type: {result['valve_type']}")
    print(f"Total calcific volume: {result['total_calcific_vol_mm3']:.1f} mm^3")
    print(f"Total fibrotic volume: {result['total_fibrotic_vol_mm3']:.1f} mm^3")
    print(f"\nPolar calcific fractions (18 zones):")
    for name, val in zip(p_names, result['polar_calcific_fractions']):
        if val > 0:
            print(f"  {name:30s} {val:.4f}")
    print(f"\nPolar fibrotic fractions (18 zones):")
    for name, val in zip(p_names, result['polar_fibrotic_fractions']):
        if val > 0:
            print(f"  {name:30s} {val:.4f}")
    print(f"\nUnwrap calcific fractions (18 zones):")
    for name, val in zip(u_names, result['unwrap_calcific_fractions']):
        if val > 0:
            print(f"  {name:30s} {val:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Tissue feature extraction: polar + unwrap projections'
    )

    parser.add_argument('--case', type=str,
                        help='Single case: compute + print features')
    parser.add_argument('--all', action='store_true',
                        help='Compute all cases')
    parser.add_argument('--compute', action='store_true',
                        help='Compute projections and cache (for batch)')
    parser.add_argument('--force', action='store_true',
                        help='Recompute even if cached')

    parser.add_argument('--seg_dir', type=str, default=str(SEGMENTATION_DIR))
    parser.add_argument('--totalseg_dir', type=str, default=str(TOTALSEG_DIR))
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR))

    args = parser.parse_args()

    seg_dir = Path(args.seg_dir)
    totalseg_dir = Path(args.totalseg_dir)
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / '.cache'
    feature_dir = output_dir / 'features'

    if args.case:
        run_single_case(args.case, seg_dir, totalseg_dir, output_dir)
        return

    if args.all or args.compute:
        run_compute(seg_dir, totalseg_dir, cache_dir, feature_dir, force=args.force)
        return

    parser.print_help()


if __name__ == '__main__':
    main()