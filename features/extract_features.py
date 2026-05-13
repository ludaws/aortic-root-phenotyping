#!/usr/bin/env python3
"""
extract_features.py - Extract geometric CT phenotyping features from aortic root segmentations.

Available masks per case (data/segmentations/<case_id>/):
    overall_mask.nii.gz         - Full aortic root wall
    root_mask.nii.gz            - Root region
    annular_mask.nii.gz         - Annular plane region
    lvot_mask.nii.gz            - LVOT region
    valve_mask.nii.gz           - Valve leaflet region
    wall_mask.nii.gz            - Wall region
    blood_pool_mask.nii.gz      - Blood pool
    cusp_R_mask.nii.gz          - Right coronary cusp sinus
    cusp_L_mask.nii.gz          - Left coronary cusp sinus
    cusp_N_mask.nii.gz          - Non-coronary cusp sinus
    calcific_mask.nii.gz        - Calcific tissue
    calcific_mask_dilated.nii.gz
    calcium_mask.nii.gz         - Calcium (thresholded)
    calcium_mask_dilated.nii.gz
    fibrotic_mask.nii.gz        - Fibrotic tissue
    tissue_params.json          - Per-case tissue classification thresholds

Available metadata.json annotations:
    patient_id                  - Case identifier
    ct_quality                  - "good" or "poor"
    valve_type                  - "tricuspid" or "type0"
    cusp_labels                 - e.g. ["right", "left", "non"]
    annulus_plane               - {center, normal} of annular plane
    annulus_nadir_points        - [3 points] one per cusp, ordered by cusp_labels
    stj_plane                   - {center, normal} of STJ plane
    lvot_plane                  - {center, normal} of LVOT plane
    commissure_points           - [3 points] between consecutive cusps
    central_coaptation_point    - Central coaptation point
    spacing                     - Voxel spacing
    origin                      - Image origin
    annulus_set                 - Whether annulus was annotated
    stj_set                     - Whether STJ was annotated
    lvot_set                    - Whether LVOT was annotated
    totalseg_file               - TotalSegmentator file path
    mpr_frame                   - MPR frame parameters

Features extracted:
    Annulus: area, area_index, ellipticity, elevation, azimuth, commissural rotation,
             rcc_nadir_angle
    Commissural plane: height, height_ratio, tilt elevation, tilt azimuth
    STJ: area, ellipticity, height, height_ratio, annulus_ratio, tilt elevation,
         tilt azimuth
    Root: volume_index, left/right/non sinus fractions
    LVOT: area, annulus_ratio, ellipticity, offset_ratio, offset_direction

Usage:
    python extract_features.py --seg_dir data/segmentations --csv data/cohort.csv \\
        --output results/phenotyping/ct_features.csv

Requires: numpy, scipy, SimpleITK, pandas
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.spatial import ConvexHull

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LVOT_DEPTH_MM = 4.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def unit(v):
    """Return unit vector."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ---------------------------------------------------------------------------
# Coordinate convention helpers (see COORDINATE_CONVENTIONS.md)
# ---------------------------------------------------------------------------
# LPS anatomical direction vectors
_DIR_ANTERIOR = np.array([0.0, -1.0, 0.0])
_DIR_RIGHT    = np.array([-1.0, 0.0, 0.0])


def enforce_cranial(normal, reference_normal):
    """Ensure *normal* points cranially (same hemisphere as *reference_normal*).

    Returns the (possibly flipped) unit normal. Logs a warning if a flip
    was needed.  See COORDINATE_CONVENTIONS.md Section 2.
    """
    n = unit(np.asarray(normal, dtype=float))
    ref = unit(np.asarray(reference_normal, dtype=float))
    if np.dot(n, ref) < 0:
        log.warning("Normal was caudal-pointing; flipped to cranial.")
        n = -n
    return n


def body_azimuth(vector_3d):
    """Azimuth of a 3D vector projected onto the CT axial plane.

    Returns degrees in [0, 360).  0 = Anterior, 90 = patient Right,
    180 = Posterior, 270 = patient Left.  Clockwise in the surgeon's
    craniocaudal view.  See COORDINATE_CONVENTIONS.md Section 4.
    """
    v = np.asarray(vector_3d, dtype=float)
    proj_ant = np.dot(v, _DIR_ANTERIOR)
    proj_rt  = np.dot(v, _DIR_RIGHT)
    return float(np.degrees(np.arctan2(proj_rt, proj_ant)) % 360)


def valve_azimuth(vector_3d, e1, e2, e3):
    """Azimuth of a 3D vector in the annular (valve) frame.

    Projects *vector_3d* onto the annular plane then returns the angle
    from e1, clockwise in the surgeon's craniocaudal view, in degrees
    [0, 360).  See COORDINATE_CONVENTIONS.md Section 6.
    """
    v = np.asarray(vector_3d, dtype=float)
    v_proj = v - np.dot(v, e3) * e3
    if np.linalg.norm(v_proj) < 1e-12:
        return np.nan
    return float(np.degrees(np.arctan2(-np.dot(v_proj, e2),
                                        np.dot(v_proj, e1))) % 360)


def cross_section_area(mask_arr, spacing_sitk, origin_sitk, plane_center, plane_normal):
    """
    Compute cross-sectional area of a binary mask at an arbitrary plane.
    Returns (area_mm2, centroid_phys) or (NaN, None) if no intersection found.
    """
    sp = np.array(spacing_sitk)
    orig = np.array(origin_sitk)
    n = unit(np.asarray(plane_normal))
    c = np.asarray(plane_center)

    zz, yy, xx = np.where(mask_arr > 0)
    if len(xx) == 0:
        return np.nan, None

    phys = np.column_stack([
        orig[0] + xx * sp[0],
        orig[1] + yy * sp[1],
        orig[2] + zz * sp[2],
    ])

    dists = (phys - c) @ n
    half_thick = 0.6 * max(sp)
    near = np.abs(dists) <= half_thick

    if np.sum(near) < 3:
        return np.nan, None

    pts_near = phys[near]
    centroid_3d = pts_near.mean(axis=0)

    if abs(n[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    u1 = unit(np.cross(n, ref))
    u2 = np.cross(n, u1)

    rel = pts_near - c
    coords_2d = np.column_stack([rel @ u1, rel @ u2])

    if len(coords_2d) < 3:
        return np.nan, None

    try:
        hull = ConvexHull(coords_2d)
        area = hull.volume
    except Exception:
        return np.nan, None

    return area, centroid_3d


def fit_ellipse_2d(points_2d):
    """
    Fit ellipse to 2D points using PCA.
    Returns (major_axis_length, minor_axis_length).
    """
    if len(points_2d) < 3:
        return np.nan, np.nan
    pts = np.asarray(points_2d)
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    major = 2.0 * np.sqrt(max(eigvals[0], 0))
    minor = 2.0 * np.sqrt(max(eigvals[1], 0))
    return major, minor


def contour_features(mask_arr, spacing_sitk, origin_sitk, plane_center, plane_normal):
    """
    Extract area, ellipticity, and average diameter from overall_mask
    cross-section at a plane.
    Returns (area_mm2, ellipticity, avg_diameter_mm).
    """
    sp = np.array(spacing_sitk)
    orig = np.array(origin_sitk)
    n = unit(np.asarray(plane_normal))
    c = np.asarray(plane_center)

    zz, yy, xx = np.where(mask_arr > 0)
    if len(xx) == 0:
        return np.nan, np.nan, np.nan

    phys = np.column_stack([
        orig[0] + xx * sp[0],
        orig[1] + yy * sp[1],
        orig[2] + zz * sp[2],
    ])

    dists = (phys - c) @ n
    half_thick = 0.6 * max(sp)
    near = np.abs(dists) <= half_thick

    if np.sum(near) < 3:
        return np.nan, np.nan, np.nan

    pts_near = phys[near]

    if abs(n[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    u1 = unit(np.cross(n, ref))
    u2 = np.cross(n, u1)

    rel = pts_near - c
    coords_2d = np.column_stack([rel @ u1, rel @ u2])

    try:
        hull = ConvexHull(coords_2d)
        area = hull.volume
    except Exception:
        return np.nan, np.nan, np.nan

    major, minor = fit_ellipse_2d(coords_2d)
    if major > 1e-6:
        ellipticity = 1.0 - (minor / major)
    else:
        ellipticity = np.nan

    # Average diameter: area-equivalent circular diameter
    avg_diam = 2.0 * np.sqrt(area / np.pi) if area > 0 else np.nan

    return area, ellipticity, avg_diam


def annular_diameter_from_area(area_mm2):
    """Effective circular diameter from area: d = 2 * sqrt(area / pi)."""
    if np.isnan(area_mm2) or area_mm2 <= 0:
        return np.nan
    return 2.0 * np.sqrt(area_mm2 / np.pi)


# ---------------------------------------------------------------------------
# Coordinate frame
# ---------------------------------------------------------------------------

def build_annulus_frame(meta):
    """
    Build the annulus local coordinate system (e1, e2, e3).

    e3 = annulus normal (cranial direction)
    e1 = annulus centroid -> RCC nadir, projected onto annulus plane
    e2 = cross(e3, e1)

    For type0 bicuspids: e1 points toward the right-equivalent cusp nadir
    (cusp label index 0).

    See COORDINATE_CONVENTIONS.md Section 5.

    Returns (e1, e2, e3, ann_center).
    """
    ann_center = np.array(meta["annulus_plane"]["center"])
    ann_normal = unit(np.array(meta["annulus_plane"]["normal"]))
    e3 = ann_normal

    valve_type = meta.get("valve_type", "tricuspid")
    cusp_labels = meta.get("cusp_labels", [])
    nadir_pts = meta.get("annulus_nadir_points")

    if nadir_pts is None or len(nadir_pts) == 0:
        log.warning("No annulus_nadir_points in metadata")
        return None, None, e3, ann_center

    if valve_type == "type0":
        # Right-equivalent cusp nadir is index 0 (anterior/right cusp)
        ref_pt = np.array(nadir_pts[0])
    else:
        # Find the "right" cusp nadir
        rcc_idx = None
        for i, lbl in enumerate(cusp_labels):
            if lbl == "right":
                rcc_idx = i
                break
        if rcc_idx is None or rcc_idx >= len(nadir_pts):
            log.warning("Could not find RCC nadir from cusp_labels: %s", cusp_labels)
            return None, None, e3, ann_center
        ref_pt = np.array(nadir_pts[rcc_idx])

    v = ref_pt - ann_center
    v_proj = v - np.dot(v, e3) * e3

    if np.linalg.norm(v_proj) < 1e-6:
        log.warning("RCC nadir coincides with annular centroid on the plane")
        return None, None, e3, ann_center

    e1 = unit(v_proj)
    e2 = unit(np.cross(e3, e1))

    return e1, e2, e3, ann_center


def commissural_plane(meta):
    """
    Compute the plane defined by the 3 commissure points.
    Returns (center, normal) or (None, None) for type0.
    """
    comm_pts = [np.array(p) for p in meta["commissure_points"]]
    if len(comm_pts) < 3:
        return None, None

    c = np.mean(comm_pts, axis=0)
    v1 = comm_pts[1] - comm_pts[0]
    v2 = comm_pts[2] - comm_pts[0]
    n = unit(np.cross(v1, v2))

    ann_n = unit(np.array(meta["annulus_plane"]["normal"]))
    if np.dot(n, ann_n) < 0:
        n = -n

    return c, n


# ---------------------------------------------------------------------------
# Feature extraction for a single case
# ---------------------------------------------------------------------------

def extract_case_features(case_dir, bsa):
    """
    Extract 20 geometric features for a single case.

    Parameters
    ----------
    case_dir : Path
        Directory containing masks and metadata.json
    bsa : float
        Body surface area (m^2)

    Returns
    -------
    dict with feature values (NaN for missing/inapplicable)
    """
    meta_path = case_dir / "metadata.json"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    if meta.get("ct_quality") == "poor" or not meta.get("annulus_set"):
        return None

    valve_type = meta.get("valve_type", "tricuspid")
    is_type0 = valve_type == "type0"
    cusp_labels = meta.get("cusp_labels", ["right", "left", "non"])

    # Load overall mask
    def load_mask(name):
        p = case_dir / name
        if not p.exists():
            return None, None, None
        img = sitk.ReadImage(str(p))
        arr = sitk.GetArrayFromImage(img)
        return arr, img.GetSpacing(), img.GetOrigin()

    overall_arr, sp_sitk, orig_sitk = load_mask("overall_mask.nii.gz")
    if overall_arr is None:
        return None

    # Load cusp masks
    if is_type0:
        cusp_mask_names = {"anterior": "cusp_A_mask.nii.gz", "posterior": "cusp_P_mask.nii.gz"}
    else:
        cusp_mask_names = {
            "left": "cusp_L_mask.nii.gz",
            "right": "cusp_R_mask.nii.gz",
            "non": "cusp_N_mask.nii.gz",
        }

    cusp_arrs = {}
    for label_name, fname in cusp_mask_names.items():
        arr, _, _ = load_mask(fname)
        cusp_arrs[label_name] = arr

    sp = np.array(sp_sitk)
    voxel_vol = sp[0] * sp[1] * sp[2]

    # Build coordinate frame (see COORDINATE_CONVENTIONS.md Section 5)
    e1, e2, e3, ann_center = build_annulus_frame(meta)
    has_frame = e1 is not None
    ann_normal = unit(np.array(meta["annulus_plane"]["normal"]))

    # Enforce cranial normal (Section 2)
    ann_normal = enforce_cranial(ann_normal, np.array([0.0, 0.0, 1.0]))

    features = {}

    # -----------------------------------------------------------------------
    # 1. Annulus (5 features)
    # -----------------------------------------------------------------------
    # Features 1-2: Annular area index and ellipticity
    ann_area, ann_ellipticity, ann_avg_diam = contour_features(
        overall_arr, sp_sitk, orig_sitk,
        meta["annulus_plane"]["center"], ann_normal
    )
    features["annular_area"] = ann_area
    features["annular_area_index"] = ann_area / bsa if not np.isnan(ann_area) else np.nan
    features["annular_ellipticity"] = ann_ellipticity
    features["annular_avg_diameter"] = ann_avg_diam
    features["_annular_area_raw"] = ann_area  # raw area for tissue volume indexing

    # Annular diameter for height normalization
    ann_diam = annular_diameter_from_area(ann_area)

    # Feature 3: Annular elevation (angle of normal from z-axis)
    nz = ann_normal[2]
    features["annular_elevation"] = np.degrees(np.arccos(np.clip(abs(nz), 0, 1)))

    # Feature 4: Annular azimuth (Section 4)
    features["annular_azimuth"] = body_azimuth(ann_normal)

    # -----------------------------------------------------------------------
    # 2. Commissural Plane (3 features) - NaN for type0
    # -----------------------------------------------------------------------
    comm_center, comm_normal = commissural_plane(meta)

    if comm_center is not None and has_frame:
        # Commissural area, ellipticity, and average diameter
        comm_area, comm_ellipticity, comm_avg_diam = contour_features(
            overall_arr, sp_sitk, orig_sitk, comm_center, comm_normal
        )
        features["comm_area"] = comm_area
        features["comm_ellipticity"] = comm_ellipticity
        features["comm_avg_diameter"] = comm_avg_diam

        # Feature 6: Commissural height ratio (height / annular diameter)
        comm_height = np.linalg.norm(comm_center - ann_center)
        features["comm_height"] = comm_height
        features["comm_height_ratio"] = comm_height / ann_diam if not np.isnan(ann_diam) and ann_diam > 0 else np.nan

        # Feature 7: Commissural tilt elevation
        comm_normal = enforce_cranial(comm_normal, ann_normal)
        dot_cn = np.clip(np.dot(comm_normal, ann_normal), -1, 1)
        features["comm_tilt_elevation"] = np.degrees(np.arccos(abs(dot_cn)))

        # Feature 8: Commissural tilt azimuth (Section 6)
        features["comm_tilt_azimuth"] = valve_azimuth(comm_normal, e1, e2, e3)
    else:
        features["comm_area"] = np.nan
        features["comm_ellipticity"] = np.nan
        features["comm_avg_diameter"] = np.nan
        features["comm_height"] = np.nan
        features["comm_height_ratio"] = np.nan
        features["comm_tilt_elevation"] = np.nan
        features["comm_tilt_azimuth"] = np.nan

    # -----------------------------------------------------------------------
    # 3. STJ (4 features)
    # -----------------------------------------------------------------------
    if meta.get("stj_set") and "stj_plane" in meta:
        stj_center = np.array(meta["stj_plane"]["center"])
        stj_normal = unit(np.array(meta["stj_plane"]["normal"]))

        # Enforce cranial normal (Section 2)
        stj_normal = enforce_cranial(stj_normal, ann_normal)

        # Feature 9: STJ/annulus area ratio
        stj_area, _ = cross_section_area(
            overall_arr, sp_sitk, orig_sitk, stj_center, stj_normal
        )
        features["stj_area"] = stj_area
        if not np.isnan(ann_area) and not np.isnan(stj_area) and ann_area > 0:
            features["stj_annulus_ratio"] = stj_area / ann_area
        else:
            features["stj_annulus_ratio"] = np.nan

        # STJ ellipticity and average diameter (from contour fitting at STJ plane)
        _, stj_ellipticity, stj_avg_diam = contour_features(
            overall_arr, sp_sitk, orig_sitk, stj_center, stj_normal
        )
        features["stj_ellipticity"] = stj_ellipticity
        features["stj_avg_diameter"] = stj_avg_diam

        # Feature 10: STJ height ratio (height / annular diameter)
        stj_height = np.linalg.norm(stj_center - ann_center)
        features["stj_height"] = stj_height
        features["stj_height_ratio"] = stj_height / ann_diam if not np.isnan(ann_diam) and ann_diam > 0 else np.nan

        # Feature 11: STJ tilt elevation
        dot_sn = np.clip(np.dot(stj_normal, ann_normal), -1, 1)
        features["stj_tilt_elevation"] = np.degrees(np.arccos(abs(dot_sn)))

        # Feature 12: STJ tilt azimuth (Section 6)
        if has_frame:
            features["stj_tilt_azimuth"] = valve_azimuth(stj_normal, e1, e2, e3)
        else:
            features["stj_tilt_azimuth"] = np.nan
    else:
        features["stj_area"] = np.nan
        features["stj_ellipticity"] = np.nan
        features["stj_avg_diameter"] = np.nan
        features["stj_height"] = np.nan
        features["stj_annulus_ratio"] = np.nan
        features["stj_height_ratio"] = np.nan
        features["stj_tilt_elevation"] = np.nan
        features["stj_tilt_azimuth"] = np.nan

    # -----------------------------------------------------------------------
    # 4. Root (4 features)
    # -----------------------------------------------------------------------
    # Feature 13: Root volume index
    root_vol = sum(np.sum(arr > 0) for arr in cusp_arrs.values() if arr is not None) * voxel_vol
    features["root_volume_index"] = root_vol / bsa

    # Features 14-16: Individual sinus fractions (cusp volume / root volume)
    if is_type0:
        for label_name in ["anterior", "posterior"]:
            arr = cusp_arrs.get(label_name)
            if arr is not None and root_vol > 0:
                features[f"{label_name}_sinus_fraction"] = (np.sum(arr > 0) * voxel_vol) / root_vol
            else:
                features[f"{label_name}_sinus_fraction"] = np.nan
        features["missing_sinus_fraction"] = np.nan
    else:
        for label_name in ["left", "right", "non"]:
            arr = cusp_arrs.get(label_name)
            if arr is not None and root_vol > 0:
                features[f"{label_name}_sinus_fraction"] = (np.sum(arr > 0) * voxel_vol) / root_vol
            else:
                features[f"{label_name}_sinus_fraction"] = np.nan

    # -----------------------------------------------------------------------
    # 5. LVOT (5 features)
    # -----------------------------------------------------------------------
    # LVOT plane = 4mm below annulus along annulus normal
    if has_frame:
        lvot_center = ann_center - LVOT_DEPTH_MM * e3
    else:
        lvot_center = ann_center - LVOT_DEPTH_MM * ann_normal
    lvot_normal = ann_normal

    # Features 17-18: LVOT/annulus ratio
    lvot_area, lvot_centroid = cross_section_area(
        overall_arr, sp_sitk, orig_sitk, lvot_center, lvot_normal
    )
    features["lvot_area"] = lvot_area

    if not np.isnan(lvot_area) and not np.isnan(ann_area) and ann_area > 0:
        features["lvot_annulus_ratio"] = lvot_area / ann_area
    else:
        features["lvot_annulus_ratio"] = np.nan

    # Feature 19: LVOT ellipticity and average diameter
    _, lvot_ellipticity, lvot_avg_diam = contour_features(
        overall_arr, sp_sitk, orig_sitk, lvot_center, lvot_normal
    )
    features["lvot_ellipticity"] = lvot_ellipticity
    features["lvot_avg_diameter"] = lvot_avg_diam

    # Features 20-21: LVOT offset magnitude ratio and direction
    if lvot_centroid is not None and has_frame:
        lvot_disp = lvot_centroid - ann_center
        lvot_disp_plane = lvot_disp - np.dot(lvot_disp, e3) * e3
        offset_mag = np.linalg.norm(lvot_disp_plane)
        features["lvot_offset_ratio"] = offset_mag / ann_diam if not np.isnan(ann_diam) and ann_diam > 0 else np.nan
        # LVOT offset direction (Section 6)
        features["lvot_offset_direction"] = valve_azimuth(lvot_disp, e1, e2, e3)
    else:
        features["lvot_offset_ratio"] = np.nan
        features["lvot_offset_direction"] = np.nan

    # -----------------------------------------------------------------------
    # 6. RCC nadir angle (Section 7)
    # -----------------------------------------------------------------------
    # Measured on the annular plane: angle from patient anterior to RCC nadir.
    # 0 = anterior, positive = clockwise from surgeon's craniocaudal view
    # (toward patient right).
    features["rcc_nadir_angle"] = np.nan
    nadir_pts = meta.get("annulus_nadir_points")
    if nadir_pts is not None and has_frame:
        rcc_idx = None
        for i, lbl in enumerate(cusp_labels):
            if lbl == "right":
                rcc_idx = i
                break
        if rcc_idx is not None and rcc_idx < len(nadir_pts):
            rcc_nadir = np.array(nadir_pts[rcc_idx])
            nadir_disp = rcc_nadir - ann_center
            nadir_plane = nadir_disp - np.dot(nadir_disp, e3) * e3

            body_ant = np.array([0.0, -1.0, 0.0])
            body_ant_plane = body_ant - np.dot(body_ant, e3) * e3
            if np.linalg.norm(body_ant_plane) > 1e-6 and np.linalg.norm(nadir_plane) > 1e-6:
                body_ant_plane_u = unit(body_ant_plane)
                nadir_plane_u = unit(nadir_plane)
                cos_a = np.clip(np.dot(nadir_plane_u, body_ant_plane_u), -1, 1)
                angle_deg = np.degrees(np.arccos(cos_a))
                # Sign: cross product with e3 determines handedness.
                # Positive cross component along e3 = CCW from anterior in
                # craniocaudal view = CW from anterior in surgeon's view
                # = toward patient right = positive by convention.
                cross_sign = np.dot(np.cross(body_ant_plane_u, nadir_plane_u), e3)
                if cross_sign < 0:
                    angle_deg = -angle_deg
                features["rcc_nadir_angle"] = angle_deg

    return features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract 76 CT phenotyping features (20 geometric + 56 tissue)")
    parser.add_argument("--seg_dir", required=True, help="Path to segmentations directory")
    parser.add_argument("--csv", required=True, help="Cohort CSV with patient covariates (BSA, valve type, etc.)")
    parser.add_argument("--tissue_dir", default="results/tissue_features/features",
                        help="Path to tissue feature JSONs from tissue_features.py")
    parser.add_argument("--output", default="results/phenotyping/ct_features.csv",
                        help="Output CSV path")
    parser.add_argument("--n_cases", type=int, default=None,
                        help="Process only first N cases (for testing)")
    args = parser.parse_args()

    seg_dir = Path(args.seg_dir)
    tissue_dir = Path(args.tissue_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load clinical data for BSA
    log.info("Loading clinical data from %s", args.csv)
    df_clin = pd.read_csv(args.csv)

    mrn_to_bsa = {}

    if "mrn" not in df_clin.columns:
        log.error("Cannot find 'mrn' column in CSV. Available: %s", list(df_clin.columns))
        sys.exit(1)

    if "Height" not in df_clin.columns or "Weight" not in df_clin.columns:
        log.error("Cannot find Height/Weight columns. Available: %s", list(df_clin.columns))
        sys.exit(1)

    for _, row in df_clin.iterrows():
        mrn = row["mrn"]
        if pd.isna(mrn):
            continue
        mrn_str = str(int(mrn)).lower() if isinstance(mrn, float) else str(mrn).lower()

        ht = row["Height"]
        wt = row["Weight"]
        if pd.notna(ht) and pd.notna(wt):
            try:
                ht_val = float(ht)
                wt_val = float(wt)
                if ht_val > 0 and wt_val > 0:
                    mrn_to_bsa[mrn_str] = np.sqrt(ht_val * wt_val / 3600.0)
            except (ValueError, TypeError):
                pass

    log.info("BSA lookup built for %d MRNs", len(mrn_to_bsa))

    # Enumerate cases
    case_dirs = sorted([d for d in seg_dir.iterdir() if d.is_dir()])
    if args.n_cases:
        case_dirs = case_dirs[:args.n_cases]

    log.info("Processing %d case directories", len(case_dirs))

    all_features = []
    skipped = 0
    failed = 0

    for i, case_dir in enumerate(case_dirs):
        case_name = case_dir.name

        meta_path = case_dir / "metadata.json"
        if not meta_path.exists():
            skipped += 1
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        if meta.get("ct_quality") == "poor" or not meta.get("annulus_set"):
            skipped += 1
            continue

        mrn = case_name.split("_")[0].lower()
        bsa = mrn_to_bsa.get(mrn)
        if bsa is None:
            log.debug("No BSA for MRN %s (%s), skipping", mrn, case_name)
            skipped += 1
            continue

        valve_type = meta.get("valve_type", "tricuspid")

        try:
            feats = extract_case_features(case_dir, bsa)
            if feats is None:
                skipped += 1
                continue

            feats["case_name"] = case_name
            feats["valve_type"] = valve_type
            feats["bsa"] = bsa
            feats["_annular_area_raw"] = feats.get("_annular_area_raw", np.nan)
            all_features.append(feats)

        except Exception as ex:
            log.error("FAILED %s: %s", case_name, ex)
            failed += 1
            continue

        if (i + 1) % 100 == 0:
            log.info("  %d / %d processed", i + 1, len(case_dirs))

    log.info("Extraction complete: %d processed, %d skipped, %d failed",
             len(all_features), skipped, failed)

    # Load tissue features if available
    tissue_available = tissue_dir.exists()
    if tissue_available:
        log.info("Loading tissue features from %s", tissue_dir)
    else:
        log.info("No tissue features directory found at %s, outputting geometric features only", tissue_dir)

    n_tissue_found = 0

    # Build DataFrame with consistent column ordering
    unified = []
    for f in all_features:
        row = {
            "case_name": f["case_name"],
            "valve_type": f["valve_type"],
            "bsa": f["bsa"],
            "annular_area": f["annular_area"],
            "annular_area_index": f["annular_area_index"],
            "annular_ellipticity": f["annular_ellipticity"],
            "annular_avg_diameter": f["annular_avg_diameter"],
            "annular_elevation": f["annular_elevation"],
            "annular_azimuth": f["annular_azimuth"],
            "comm_area": f["comm_area"],
            "comm_ellipticity": f["comm_ellipticity"],
            "comm_avg_diameter": f["comm_avg_diameter"],
            "comm_height": f["comm_height"],
            "comm_height_ratio": f["comm_height_ratio"],
            "comm_tilt_elevation": f["comm_tilt_elevation"],
            "comm_tilt_azimuth": f["comm_tilt_azimuth"],
            "stj_area": f["stj_area"],
            "stj_ellipticity": f["stj_ellipticity"],
            "stj_avg_diameter": f["stj_avg_diameter"],
            "stj_height": f["stj_height"],
            "stj_annulus_ratio": f["stj_annulus_ratio"],
            "stj_height_ratio": f["stj_height_ratio"],
            "stj_tilt_elevation": f["stj_tilt_elevation"],
            "stj_tilt_azimuth": f["stj_tilt_azimuth"],
            "root_volume_index": f["root_volume_index"],
            "rcc_nadir_angle": f["rcc_nadir_angle"],
        }

        # Sinus fractions: map type0 anterior/posterior to consistent columns
        if f["valve_type"] == "type0":
            row["left_sinus_fraction"] = f.get("anterior_sinus_fraction", np.nan)
            row["right_sinus_fraction"] = f.get("posterior_sinus_fraction", np.nan)
            row["non_sinus_fraction"] = np.nan
        else:
            row["left_sinus_fraction"] = f.get("left_sinus_fraction", np.nan)
            row["right_sinus_fraction"] = f.get("right_sinus_fraction", np.nan)
            row["non_sinus_fraction"] = f.get("non_sinus_fraction", np.nan)

        row.update({
            "lvot_area": f["lvot_area"],
            "lvot_avg_diameter": f["lvot_avg_diameter"],
            "lvot_annulus_ratio": f["lvot_annulus_ratio"],
            "lvot_ellipticity": f["lvot_ellipticity"],
            "lvot_offset_ratio": f["lvot_offset_ratio"],
            "lvot_offset_direction": f["lvot_offset_direction"],
        })

        # Load tissue features for this case
        if tissue_available:
            tissue_path = tissue_dir / f"{f['case_name']}_tissue_features.json"
            if tissue_path.exists():
                with open(tissue_path) as tf:
                    tissue = json.load(tf)
                n_tissue_found += 1

                # Absolute volumes (mm3)
                row["calcific_volume_mm3"] = tissue["total_calcific_vol_mm3"]
                row["fibrotic_volume_mm3"] = tissue["total_fibrotic_vol_mm3"]

                # Index total volumes to annular area (mm3 / cm2)
                ann_area_raw = f.get("_annular_area_raw")
                ann_area_cm2 = ann_area_raw / 100.0 if ann_area_raw and not np.isnan(ann_area_raw) else None

                if ann_area_cm2 and ann_area_cm2 > 0:
                    row["calcific_volume_index"] = tissue["total_calcific_vol_mm3"] / ann_area_cm2
                    row["fibrotic_volume_index"] = tissue["total_fibrotic_vol_mm3"] / ann_area_cm2

                    # Per-cusp indexed volumes (mm3 / cm2)
                    cusp_calc = tissue.get("cusp_calcific_vols_mm3", {})
                    cusp_fibr = tissue.get("cusp_fibrotic_vols_mm3", {})

                    # Tricuspid cusps
                    for cusp_name in ["right", "left", "non"]:
                        row[f"calcific_{cusp_name}_index"] = cusp_calc.get(cusp_name, np.nan) / ann_area_cm2 if cusp_calc.get(cusp_name) is not None else np.nan
                        row[f"fibrotic_{cusp_name}_index"] = cusp_fibr.get(cusp_name, np.nan) / ann_area_cm2 if cusp_fibr.get(cusp_name) is not None else np.nan
                        row[f"calcific_{cusp_name}_mm3"] = cusp_calc.get(cusp_name, np.nan) if cusp_calc.get(cusp_name) is not None else np.nan
                        row[f"fibrotic_{cusp_name}_mm3"] = cusp_fibr.get(cusp_name, np.nan) if cusp_fibr.get(cusp_name) is not None else np.nan

                    # Type0 cusps (anterior/posterior)
                    for cusp_name in ["anterior", "posterior"]:
                        val_c = cusp_calc.get(cusp_name)
                        val_f = cusp_fibr.get(cusp_name)
                        row[f"calcific_{cusp_name}_index"] = val_c / ann_area_cm2 if val_c is not None else np.nan
                        row[f"fibrotic_{cusp_name}_index"] = val_f / ann_area_cm2 if val_f is not None else np.nan
                        row[f"calcific_{cusp_name}_mm3"] = val_c if val_c is not None else np.nan
                        row[f"fibrotic_{cusp_name}_mm3"] = val_f if val_f is not None else np.nan
                else:
                    row["calcific_volume_index"] = np.nan
                    row["fibrotic_volume_index"] = np.nan
                    for cusp_name in ["right", "left", "non", "anterior", "posterior"]:
                        row[f"calcific_{cusp_name}_index"] = np.nan
                        row[f"fibrotic_{cusp_name}_index"] = np.nan

        unified.append(row)

    if tissue_available:
        log.info("Tissue features loaded for %d / %d cases", n_tissue_found, len(all_features))

    # Build column list
    geometric_cols = [
        "case_name", "valve_type", "bsa",
        # Annulus
        "annular_area", "annular_area_index", "annular_ellipticity",
        "annular_avg_diameter",
        "annular_elevation", "annular_azimuth",
        "rcc_nadir_angle",
        # Commissural plane
        "comm_area", "comm_ellipticity", "comm_avg_diameter",
        "comm_height", "comm_height_ratio", "comm_tilt_elevation", "comm_tilt_azimuth",
        # STJ
        "stj_area", "stj_ellipticity", "stj_avg_diameter",
        "stj_height",
        "stj_annulus_ratio", "stj_height_ratio", "stj_tilt_elevation", "stj_tilt_azimuth",
        # Root
        "root_volume_index", "left_sinus_fraction", "right_sinus_fraction",
        "non_sinus_fraction",
        # LVOT
        "lvot_area", "lvot_avg_diameter",
        "lvot_annulus_ratio", "lvot_ellipticity",
        "lvot_offset_ratio", "lvot_offset_direction",
    ]

    # Discover tissue columns from first case that has them
    tissue_cols = []
    if tissue_available and n_tissue_found > 0:
        for row in unified:
            if "calcific_volume_index" in row:
                tissue_cols = [k for k in row.keys() if k not in geometric_cols]
                break

    feature_cols = geometric_cols + tissue_cols

    df = pd.DataFrame(unified)
    # Ensure all columns exist (NaN for cases missing tissue features)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[feature_cols]

    df.to_csv(output_path, index=False, float_format="%.4f")
    log.info("Saved %d cases x %d columns (%d geometric + %d tissue) to %s",
             len(df), len(feature_cols), len(geometric_cols), len(tissue_cols), output_path)

    # Summary statistics
    log.info("\n--- Feature Summary ---")
    numeric_cols = [c for c in feature_cols if c not in ("case_name", "valve_type")]
    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) > 0:
            log.info("  %-30s  n=%4d  median=%.2f  [%.2f - %.2f]",
                     col, len(vals), vals.median(),
                     vals.quantile(0.25), vals.quantile(0.75))

    # Log metadata
    log_data = {
        "n_processed": len(df),
        "n_skipped": skipped,
        "n_failed": failed,
        "feature_columns": feature_cols,
    }
    log_path = output_path.with_suffix(".json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    log.info("Saved run metadata to %s", log_path)


if __name__ == "__main__":
    main()