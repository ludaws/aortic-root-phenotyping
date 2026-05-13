#!/usr/bin/env python3
"""
Aortic Root MPR Segmentation Tool.

Interactive multi-planar reformatting (MPR) annotator for placement of:

  - Three cusp nadir points (right, left, non-coronary)
  - Central coaptation point and three commissural points
  - Annular, sino-tubular junction (STJ), and LVOT planes
  - Sievers bicuspid type and per-case CT quality flag

Outputs per case:

  metadata.json            - all annotation parameters
  overall_mask.nii.gz      - combined root + LVOT
  lvot_mask.nii.gz         - LVOT region (annulus to LVOT plane)
  cusp_R / cusp_L / cusp_N
  root_mask.nii.gz         - root cusps (annulus to STJ plane)

Either DICOM folders or de-identified NPZ volumes may be used as input
(auto-detected from the contents of data_dir).

Interaction:

  Translation: drag crosshair centre to translate the active view.
  Rotation:    drag crosshair line ends to rotate the active view; the
               other two MPR views update to maintain orthogonality.

Usage:

    python aortic_segmentator.py \\
        --data_dir <path/to/dicom_or_npz_dir> \\
        --totalseg_dir <path/to/totalseg_outputs> \\
        --output_dir <path/to/output>

Optional flags allow exclusion of cases listed in a CSV (e.g. prior aortic
valve replacement) via --csv, --id_column, --exclude_column, --exclude_value.
"""

import argparse
import json
import logging
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
try:
    import pydicom
except ImportError:
    pydicom = None
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.gridspec as gridspec
import SimpleITK as sitk
from geomdl import BSpline as GeomdlBSpline
from geomdl import exchange as geomdl_exchange
from geomdl import utilities as geomdl_utilities

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

PANEL_COLORS = ['red', 'green', 'blue']
PANEL_NAMES = ['Sagittal', 'Coronal', 'Axial']
ROTATION_SIGN = [-1, 1, -1]

COLORS = {
    'bg': '#1E1E1E',
    'panel_bg': '#2D2D2D',
    'button': '#3A3A3A',
    'button_hover': '#4A4A4A',
    'active': '#0D7377',
    'set_unset': '#555555',
    'set_done': '#4CAF50',
    'text': '#E0E0E0',
    'text_dim': '#888888',
    'poor_quality': '#D32F2F',
    'acceptable_quality': '#4CAF50',
    'totalseg_toggle_on': '#1976D2',
    'totalseg_toggle_off': '#555555',
}

# TotalSegmentator label mapping for heartchambers_highres task
TOTALSEG_LABELS = {
    'myocardium': 1,
    'atrium_left': 2,
    'ventricle_left': 3,
    'atrium_right': 4,
    'ventricle_right': 5,
    'aorta': 6,
    'pulmonary_artery': 7,
}

# Colors for TotalSegmentator overlay (RGBA)
TOTALSEG_COLORS = {
    3: (1.0, 0.0, 0.0, 0.25),    # ventricle_left - red
    6: (0.0, 0.5, 1.0, 0.25),    # aorta - blue
}


@dataclass
class SegmentationPlane:
    """
    Defines a plane in 3D DICOM physical coordinates.
    
    A plane is fully defined by:
    - center: A point on the plane [x, y, z] in mm
    - normal: Unit normal vector [nx, ny, nz]
    
    This representation is independent of any view state and can be
    directly used for segmentation and export.
    """
    center: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    normal: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    
    def to_dict(self):
        return {
            'center': self.center,
            'normal': self.normal,
        }
    
    @classmethod
    def from_dict(cls, d):
        return cls(center=d['center'], normal=d['normal'])
    
    def get_normal_array(self) -> np.ndarray:
        return np.array(self.normal)
    
    def get_center_array(self) -> np.ndarray:
        return np.array(self.center)


class BSplineBoundarySurface:
    """
    Tensor product B-spline surface for boundary representation.
    
    Two surfaces share a boundary at MS nadir:
    - ILT/MS surface: from RN commissure (degenerate point) to MS nadir
    - LVOT surface: from MS nadir to LVOT plane
    
    Each surface is defined by control points at discrete t-levels,
    with cubic B-spline interpolation in both U (around boundary) and V (along axis) directions.
    """
    
    def __init__(self, control_points, t_values, axis_center, axis_normal, axis_length):
        """
        Args:
            control_points: List of arrays, each array is [n_pts, 3] control points at that t-level
            t_values: List of t values (0 to 1) for each control point level
            axis_center: Center point of the axis (annulus center)
            axis_normal: Normal direction along axis
            axis_length: Total length along axis
        """
        self.control_points = [np.array(cp) for cp in control_points]
        self.t_values = np.array(t_values)
        self.axis_center = np.array(axis_center)
        self.axis_normal = np.array(axis_normal) / np.linalg.norm(axis_normal)
        self.axis_length = axis_length
        
        # Build 2D coordinate system perpendicular to axis
        if abs(self.axis_normal[2]) < 0.9:
            ref = np.array([0, 0, 1])
        else:
            ref = np.array([1, 0, 0])
        self.plane_x = np.cross(self.axis_normal, ref)
        self.plane_x = self.plane_x / np.linalg.norm(self.plane_x)
        self.plane_y = np.cross(self.axis_normal, self.plane_x)
        
        # Precompute splines at each t-level for fast evaluation
        self._precompute_level_splines()
    
    def _precompute_level_splines(self):
        """Precompute 2D spline (angle -> radius) at each t-level."""
        self.level_splines = []
        
        for i, cp in enumerate(self.control_points):
            axis_pt = self._get_axis_point(self.t_values[i])
            
            # Project control points to 2D
            pts_2d = []
            for pt in cp:
                d = pt - axis_pt
                pts_2d.append([np.dot(d, self.plane_x), np.dot(d, self.plane_y)])
            pts_2d = np.array(pts_2d)
            
            # Fit spline through 2D points
            if len(pts_2d) >= 4:
                try:
                    tck, u = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=0, k=3)
                    spline_pts = np.array(splev(np.linspace(0, 1, 100), tck)).T
                except:
                    spline_pts = pts_2d
            else:
                spline_pts = pts_2d
            
            # Convert to angle -> radius lookup
            angles = np.arctan2(spline_pts[:, 1], spline_pts[:, 0])
            radii = np.sqrt(spline_pts[:, 0]**2 + spline_pts[:, 1]**2)
            sort_idx = np.argsort(angles)
            
            self.level_splines.append({
                'angles': angles[sort_idx],
                'radii': radii[sort_idx],
                'angle_min': angles.min(),
                'angle_max': angles.max(),
            })
    
    def _get_axis_point(self, t):
        """Get point on central axis at parameter t."""
        return self.axis_center + t * self.axis_length * self.axis_normal
    
    def get_radius_at(self, t, angle):
        """
        Get boundary radius at parameter t and angle.
        Uses cubic interpolation between levels.
        
        Returns None if angle is outside boundary range.
        """
        # Find bracketing t-levels
        if t <= self.t_values[0]:
            idx_lo, idx_hi, frac = 0, 0, 0.0
        elif t >= self.t_values[-1]:
            idx_lo, idx_hi, frac = len(self.t_values)-1, len(self.t_values)-1, 0.0
        else:
            idx_hi = np.searchsorted(self.t_values, t)
            idx_lo = idx_hi - 1
            frac = (t - self.t_values[idx_lo]) / (self.t_values[idx_hi] - self.t_values[idx_lo])
        
        # Get radius from both levels
        sp_lo = self.level_splines[idx_lo]
        sp_hi = self.level_splines[idx_hi]
        
        # Check angle range (use intersection of ranges)
        a_min = max(sp_lo['angle_min'], sp_hi['angle_min'])
        a_max = min(sp_lo['angle_max'], sp_hi['angle_max'])
        
        # Normalize angle check
        angle_norm = (angle - a_min) % (2 * np.pi)
        span = (a_max - a_min) % (2 * np.pi)
        if angle_norm > span:
            return None
        
        # Interpolate radius at angle from both levels
        r_lo = np.interp(angle, sp_lo['angles'], sp_lo['radii'])
        r_hi = np.interp(angle, sp_hi['angles'], sp_hi['radii'])
        
        # Linear interpolation between levels
        return r_lo * (1 - frac) + r_hi * frac
    
    def get_contour_at_t(self, t, n_points=100):
        """Get boundary contour points at parameter t in physical coordinates."""
        axis_pt = self._get_axis_point(t)
        
        # Find bracketing levels and interpolate control points
        if t <= self.t_values[0]:
            cp = self.control_points[0]
        elif t >= self.t_values[-1]:
            cp = self.control_points[-1]
        else:
            idx_hi = np.searchsorted(self.t_values, t)
            idx_lo = idx_hi - 1
            frac = (t - self.t_values[idx_lo]) / (self.t_values[idx_hi] - self.t_values[idx_lo])
            
            cp_lo = self.control_points[idx_lo]
            cp_hi = self.control_points[idx_hi]
            
            # Handle different numbers of control points
            if len(cp_lo) == len(cp_hi):
                cp = cp_lo * (1 - frac) + cp_hi * frac
            else:
                # Use spline evaluation instead
                sp_lo = self.level_splines[idx_lo]
                sp_hi = self.level_splines[idx_hi]
                
                angles = np.linspace(sp_lo['angle_min'], sp_lo['angle_max'], n_points)
                r_lo = np.interp(angles, sp_lo['angles'], sp_lo['radii'])
                r_hi = np.interp(angles, sp_hi['angles'], sp_hi['radii'])
                radii = r_lo * (1 - frac) + r_hi * frac
                
                # Convert to 3D physical coordinates
                pts_3d = []
                for ang, r in zip(angles, radii):
                    pt_2d = np.array([r * np.cos(ang), r * np.sin(ang)])
                    pt_3d = axis_pt + pt_2d[0] * self.plane_x + pt_2d[1] * self.plane_y
                    pts_3d.append(pt_3d)
                return np.array(pts_3d)
        
        # Fit spline through control points
        pts_2d = []
        for pt in cp:
            d = pt - axis_pt
            pts_2d.append([np.dot(d, self.plane_x), np.dot(d, self.plane_y)])
        pts_2d = np.array(pts_2d)
        
        if len(pts_2d) >= 4:
            try:
                tck, u = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=0, k=3)
                spline_2d = np.array(splev(np.linspace(0, 1, n_points), tck)).T
            except:
                spline_2d = pts_2d
        else:
            spline_2d = pts_2d
        
        # Convert to 3D
        pts_3d = []
        for pt_2d in spline_2d:
            pt_3d = axis_pt + pt_2d[0] * self.plane_x + pt_2d[1] * self.plane_y
            pts_3d.append(pt_3d)
        
        return np.array(pts_3d)
    
    def to_dict(self):
        """Serialize to dictionary for JSON storage."""
        return {
            'control_points': [cp.tolist() for cp in self.control_points],
            't_values': self.t_values.tolist(),
            'axis_center': self.axis_center.tolist(),
            'axis_normal': self.axis_normal.tolist(),
            'axis_length': float(self.axis_length),
        }
    
    @classmethod
    def from_dict(cls, d):
        """Reconstruct from dictionary."""
        return cls(
            control_points=d['control_points'],
            t_values=d['t_values'],
            axis_center=d['axis_center'],
            axis_normal=d['axis_normal'],
            axis_length=d['axis_length'],
        )


@dataclass
class CaseState:
    """State for a single case annotation session."""
    patient_id: str = ""
    
    # CT Quality assessment
    ct_quality: str = "acceptable"  # "unset", "acceptable", "poor"
    
    # Plane definitions (stored as center + normal in physical DICOM coordinates)
    annulus_plane: Optional[SegmentationPlane] = None
    stj_plane: Optional[SegmentationPlane] = None
    lvot_plane: Optional[SegmentationPlane] = None
    
    # Annulus nadir points: [Right, Left, Non] cusp nadirs in physical DICOM coordinates
    annulus_nadir_points: Optional[List[List[float]]] = None
    
    # Valve type: "tricuspid", "type0", "type1_lr", "type1_rn", "type1_nl", "type2"
    valve_type: str = "tricuspid"
    
    # Cusp labels corresponding to the sectors between commissure points
    # For tricuspid/type1/type2: 3 labels
    # For type0: 2 labels (anterior, posterior)
    cusp_labels: List[str] = field(default_factory=lambda: ["right", "left", "non"])
    
    # CC/ILT/MS (Central Coaptation / Interleaflet Triangle / Membranous Septum) annotation
    # Central coaptation point: where leaflets come together (center for cusp division)
    central_coaptation_point: Optional[List[float]] = None
    # Commissure points: [RN, LR, NL] in physical DICOM coordinates (define cusp sector boundaries)
    commissure_points: Optional[List[List[float]]] = None
    # MS nadir point: most inferior point of membranous septum (refined to closest boundary point)
    ms_nadir_point: Optional[List[float]] = None
    # ILT/MS boundary points: 5 levels x 4 points = 20 points
    # Levels at t=0.2, 0.4, 0.6, 0.8, 0.95 (surface tapers to MS nadir at t=1.0)
    ilt_ms_points: List[List[float]] = field(default_factory=list)
    ilt_ms_set: bool = False
    
    # LVOT septal boundary points: 3 levels x 4 points = 12 points
    # Level 0 (MS nadir): 3 clicked + MS nadir auto-inserted as middle point = 4 points
    # Level 1 (midpoint at t=0.5): 4 clicked points
    # Level 2 (LVOT plane at t=1.0): 4 clicked points
    lvot_septal_boundary_points: List[List[float]] = field(default_factory=list)
    
    # Boundary surface parameters (saved for reconstruction)
    # ILT/MS surface: from RN commissure to MS nadir
    ilt_ms_surface_params: Optional[dict] = None
    # LVOT surface: from MS nadir to LVOT plane  
    lvot_surface_params: Optional[dict] = None
    
    # Volume metadata
    spacing: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    origin: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    
    # Set flags - indicate which annotations have been confirmed
    annulus_set: bool = False
    stj_set: bool = False
    lvot_set: bool = False
    
    # TotalSegmentator info
    totalseg_file: Optional[str] = None
    
    # MPR frame at time of annotation (for reproducibility)
    # This allows re-projecting annotations back to the same view
    mpr_frame: Optional[List[List[float]]] = None
    
    def to_dict(self):
        return {
            'patient_id': self.patient_id,
            'ct_quality': self.ct_quality,
            'valve_type': self.valve_type,
            'cusp_labels': self.cusp_labels,
            'annulus_plane': self.annulus_plane.to_dict() if self.annulus_plane else None,
            'annulus_nadir_points': self.annulus_nadir_points,
            'stj_plane': self.stj_plane.to_dict() if self.stj_plane else None,
            'lvot_plane': self.lvot_plane.to_dict() if self.lvot_plane else None,
            'commissure_points': self.commissure_points,
            'central_coaptation_point': self.central_coaptation_point,
            'ms_nadir_point': self.ms_nadir_point,
            'ilt_ms_points': self.ilt_ms_points,
            'ilt_ms_set': self.ilt_ms_set,
            'lvot_septal_boundary_points': self.lvot_septal_boundary_points,
            'ilt_ms_surface_params': self.ilt_ms_surface_params,
            'lvot_surface_params': self.lvot_surface_params,
            'spacing': self.spacing,
            'origin': self.origin,
            'annulus_set': self.annulus_set,
            'stj_set': self.stj_set,
            'lvot_set': self.lvot_set,
            'totalseg_file': self.totalseg_file,
            'mpr_frame': self.mpr_frame,
        }
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path):
        with open(path) as f:
            d = json.load(f)
        state = cls(
            patient_id=d['patient_id'],
            ct_quality=d.get('ct_quality', 'unset'),
            valve_type=d.get('valve_type', 'tricuspid'),
            cusp_labels=d.get('cusp_labels', ['right', 'left', 'non']),
            annulus_nadir_points=d.get('annulus_nadir_points'),
            commissure_points=d.get('commissure_points'),
            central_coaptation_point=d.get('central_coaptation_point'),
            ms_nadir_point=d.get('ms_nadir_point'),
            ilt_ms_points=d.get('ilt_ms_points', []),
            ilt_ms_set=d.get('ilt_ms_set', False),
            lvot_septal_boundary_points=d.get('lvot_septal_boundary_points', []),
            ilt_ms_surface_params=d.get('ilt_ms_surface_params'),
            lvot_surface_params=d.get('lvot_surface_params'),
            spacing=d.get('spacing'),
            origin=d.get('origin'),
            annulus_set=d.get('annulus_set', False),
            stj_set=d.get('stj_set', False),
            lvot_set=d.get('lvot_set', False),
            totalseg_file=d.get('totalseg_file'),
            mpr_frame=d.get('mpr_frame'),
        )
        if d.get('annulus_plane'):
            state.annulus_plane = SegmentationPlane.from_dict(d['annulus_plane'])
        if d.get('stj_plane'):
            state.stj_plane = SegmentationPlane.from_dict(d['stj_plane'])
        if d.get('lvot_plane'):
            state.lvot_plane = SegmentationPlane.from_dict(d['lvot_plane'])
        return state
    
    def is_poor_quality(self) -> bool:
        return self.ct_quality == "poor"
    
    def is_type0_bicuspid(self) -> bool:
        """True bicuspid (type 0) has only 2 cusps."""
        return self.valve_type == "type0"
    
    def expected_nadir_count(self) -> int:
        """Number of nadir points expected: 2 for type0, 3 otherwise."""
        return 2 if self.is_type0_bicuspid() else 3
    
    def expected_commissure_count(self) -> int:
        """Number of commissure points expected: 2 for type0, 3 otherwise."""
        return 2 if self.is_type0_bicuspid() else 3
    
    def expected_ilt_ms_point_count(self) -> int:
        """Total ILT/MS boundary points.
        
        Tricuspid: 5 levels x 4 points = 20 (from RN commissure to MS nadir)
        Type0: 4 levels x 4 points = 16 (from annulus to MS nadir, no interleaflet triangle)
        """
        return 16 if self.is_type0_bicuspid() else 20


def load_dicom_volume(case_folder: Path):
    """Load DICOM volume from a case folder. Requires pydicom."""
    if pydicom is None:
        raise ImportError(
            "pydicom is required for DICOM loading. "
            "Install it with: pip install pydicom"
        )
    dicom_files = []
    for f in case_folder.rglob('*'):
        if f.is_file() and not f.name.startswith('.'):
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            if hasattr(ds, 'Modality') and ds.Modality == 'CT':
                dicom_files.append(str(f))
    
    if not dicom_files:
        raise ValueError(f"No DICOM files in {case_folder}")
    
    slices = []
    for f in dicom_files:
        ds = pydicom.dcmread(f)
        if hasattr(ds, 'ImagePositionPatient') and hasattr(ds, 'pixel_array'):
            slices.append(ds)
    
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    ds0 = slices[0]
    pixel_spacing = [float(x) for x in ds0.PixelSpacing]
    slice_spacing = abs(float(slices[1].ImagePositionPatient[2]) - 
                       float(slices[0].ImagePositionPatient[2])) if len(slices) > 1 else 1.0
    
    spacing = np.array([slice_spacing, pixel_spacing[1], pixel_spacing[0]])
    origin = np.array([float(x) for x in ds0.ImagePositionPatient])
    
    volume = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
    slope = float(getattr(ds0, 'RescaleSlope', 1))
    intercept = float(getattr(ds0, 'RescaleIntercept', 0))
    volume = volume * slope + intercept
    
    return volume, spacing, origin


def load_npz_volume(npz_path: Path):
    """Load de-identified CT volume from an NPZ file.
    
    The NPZ 'origin' key may store a cropped origin even when the volume
    is the full uncropped array. When volume shape matches original_shape,
    original_origin is used instead.
    
    Returns:
        volume:  float32 array (Z, Y, X)
        spacing: ndarray (3,) in (z, y, x) order
        origin:  ndarray (3,) in DICOM (x, y, z) order
    """
    data = np.load(str(npz_path), allow_pickle=True)
    
    volume = data['volume'].astype(np.float32)
    spacing = np.array(data['spacing'], dtype=np.float64)
    origin = np.array(data['origin'], dtype=np.float64)
    
    # Use original_origin when the volume is full-size (not actually cropped)
    if 'original_origin' in data and 'original_shape' in data:
        original_shape = np.array(data['original_shape'])
        if np.array_equal(np.array(volume.shape), original_shape):
            origin = np.array(data['original_origin'], dtype=np.float64)
    
    logger.info(
        f"Loaded NPZ: shape={volume.shape}, "
        f"spacing=[{spacing[0]:.3f}, {spacing[1]:.3f}, {spacing[2]:.3f}] mm, "
        f"origin=[{origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f}]"
    )
    
    return volume, spacing, origin


def load_totalseg_mask(totalseg_dir: Path, case_name: str) -> Optional[Tuple[np.ndarray, dict]]:
    """Load TotalSegmentator cardiac segmentation mask."""
    case_dir = totalseg_dir / case_name
    if not case_dir.exists():
        return None
    
    crop_info = None
    crop_info_path = case_dir / 'crop_info.json'
    if crop_info_path.exists():
        with open(crop_info_path) as f:
            crop_info = json.load(f)
        logger.info(f"Found crop_info: {crop_info['original_shape']} -> {crop_info['cropped_shape']}")
    
    candidates = ['totalseg_cardiac.nii.gz', 'totalseg_combined.nii.gz']
    mask_file = None
    for candidate in candidates:
        if (case_dir / candidate).exists():
            mask_file = case_dir / candidate
            break
    
    if mask_file is None:
        aorta_file = case_dir / 'aorta.nii.gz'
        lv_file = case_dir / 'ventricle_left.nii.gz'
        
        if aorta_file.exists() or lv_file.exists():
            combined = None
            spacing = None
            origin = None
            source_files = []
            
            if lv_file.exists():
                img = sitk.ReadImage(str(lv_file))
                arr = sitk.GetArrayFromImage(img)
                combined = np.zeros_like(arr, dtype=np.uint8)
                combined[arr > 0] = TOTALSEG_LABELS['ventricle_left']
                spacing = img.GetSpacing()
                origin = img.GetOrigin()
                source_files.append('ventricle_left.nii.gz')
            
            if aorta_file.exists():
                img = sitk.ReadImage(str(aorta_file))
                arr = sitk.GetArrayFromImage(img)
                if combined is None:
                    combined = np.zeros_like(arr, dtype=np.uint8)
                    spacing = img.GetSpacing()
                    origin = img.GetOrigin()
                combined[arr > 0] = TOTALSEG_LABELS['aorta']
                source_files.append('aorta.nii.gz')
            
            if combined is not None:
                return combined, {
                    'spacing': list(spacing),
                    'origin': list(origin),
                    'crop_info': crop_info,
                    'source_files': source_files
                }
        return None
    
    img = sitk.ReadImage(str(mask_file))
    arr = sitk.GetArrayFromImage(img)
    
    return arr, {
        'spacing': list(img.GetSpacing()),
        'origin': list(img.GetOrigin()),
        'crop_info': crop_info,
        'source_file': mask_file.name
    }


class MPREngine:
    
    def __init__(self, volume, spacing, origin):
        self.volume = volume
        self.spacing = spacing
        self.origin = origin
        self.shape = np.array(volume.shape)
        
        self.phys_min = origin.copy()
        self.phys_max = np.array([
            origin[0] + (self.shape[2] - 1) * spacing[2],
            origin[1] + (self.shape[1] - 1) * spacing[1],
            origin[2] + (self.shape[0] - 1) * spacing[0]
        ])
        
        self.intersection = (self.phys_min + self.phys_max) / 2
        self.frame = np.eye(3)
        
        self.slice_width = 400
        self.slice_height = 300
        self.fov = max(self.phys_max - self.phys_min) * 0.384
        self.output_spacing = self.fov / self.slice_width
        
        self.window_center = 300
        self.window_width = 1500
    
    def copy_frame(self):
        return self.frame.copy()
    
    def get_plane_axes(self, panel_idx, frame_override=None):
        frame = frame_override if frame_override is not None else self.frame
        
        if panel_idx == 0:  # Sagittal
            normal = frame[0]
            x_axis = -frame[1]
            y_axis = frame[2]
        elif panel_idx == 1:  # Coronal
            normal = frame[1]
            x_axis = -frame[0]
            y_axis = frame[2]
        else:  # Axial
            normal = frame[2]
            x_axis = -frame[0]
            y_axis = frame[1]
        return x_axis.copy(), y_axis.copy(), normal.copy()
    
    def extract_slice(self, panel_idx, center_override=None, frame_override=None):
        center = center_override if center_override is not None else self.intersection
        x_axis, y_axis, _ = self.get_plane_axes(panel_idx, frame_override)
        
        half_w = self.slice_width // 2
        half_h = self.slice_height // 2
        u = np.linspace(-half_w, half_w, self.slice_width) * self.output_spacing
        v = np.linspace(-half_h, half_h, self.slice_height) * self.output_spacing
        uu, vv = np.meshgrid(u, v, indexing='xy')
        
        coords_phys = (center[np.newaxis, np.newaxis, :] +
                       uu[:, :, np.newaxis] * x_axis +
                       vv[:, :, np.newaxis] * y_axis)
        
        coords_voxel = np.zeros_like(coords_phys)
        coords_voxel[:, :, 0] = (coords_phys[:, :, 2] - self.origin[2]) / self.spacing[0]
        coords_voxel[:, :, 1] = (coords_phys[:, :, 1] - self.origin[1]) / self.spacing[1]
        coords_voxel[:, :, 2] = (coords_phys[:, :, 0] - self.origin[0]) / self.spacing[2]
        
        slice_data = ndimage.map_coordinates(
            self.volume,
            [coords_voxel[:, :, 0].ravel(),
             coords_voxel[:, :, 1].ravel(),
             coords_voxel[:, :, 2].ravel()],
            order=1, mode='constant', cval=-1000
        ).reshape(self.slice_height, self.slice_width)
        
        return self._apply_window(slice_data)
    
    def extract_mask_slice(self, panel_idx, mask, center_override=None, frame_override=None):
        center = center_override if center_override is not None else self.intersection
        x_axis, y_axis, _ = self.get_plane_axes(panel_idx, frame_override)
        
        half_w = self.slice_width // 2
        half_h = self.slice_height // 2
        u = np.linspace(-half_w, half_w, self.slice_width) * self.output_spacing
        v = np.linspace(-half_h, half_h, self.slice_height) * self.output_spacing
        uu, vv = np.meshgrid(u, v, indexing='xy')
        
        coords_phys = (center[np.newaxis, np.newaxis, :] +
                       uu[:, :, np.newaxis] * x_axis +
                       vv[:, :, np.newaxis] * y_axis)
        
        coords_voxel = np.zeros_like(coords_phys)
        coords_voxel[:, :, 0] = (coords_phys[:, :, 2] - self.origin[2]) / self.spacing[0]
        coords_voxel[:, :, 1] = (coords_phys[:, :, 1] - self.origin[1]) / self.spacing[1]
        coords_voxel[:, :, 2] = (coords_phys[:, :, 0] - self.origin[0]) / self.spacing[2]
        
        return ndimage.map_coordinates(
            mask.astype(float),
            [coords_voxel[:, :, 0].ravel(),
             coords_voxel[:, :, 1].ravel(),
             coords_voxel[:, :, 2].ravel()],
            order=0, mode='constant', cval=0
        ).reshape(self.slice_height, self.slice_width)
    
    def _apply_window(self, data):
        vmin = self.window_center - self.window_width / 2
        vmax = self.window_center + self.window_width / 2
        return np.clip((data - vmin) / (vmax - vmin), 0, 1)
    
    def pixel_to_physical_offset(self, panel_idx, dx_px, dy_px, frame_override=None):
        x_axis, y_axis, _ = self.get_plane_axes(panel_idx, frame_override)
        return dx_px * self.output_spacing * x_axis + dy_px * self.output_spacing * y_axis
    
    @staticmethod
    def rotate_frame_by_axis(frame, axis_vector, angle_deg):
        angle_rad = np.radians(angle_deg)
        k = axis_vector / np.linalg.norm(axis_vector)
        
        def rotate_vector(v):
            return (v * np.cos(angle_rad) + 
                    np.cross(k, v) * np.sin(angle_rad) + 
                    k * np.dot(k, v) * (1 - np.cos(angle_rad)))
        
        new_frame = np.zeros_like(frame)
        for i in range(3):
            new_frame[i] = rotate_vector(frame[i])
            new_frame[i] /= np.linalg.norm(new_frame[i])
        
        return new_frame
    
    def get_crosshair_colors(self, panel_idx):
        if panel_idx == 0:
            return PANEL_COLORS[2], PANEL_COLORS[1]
        elif panel_idx == 1:
            return PANEL_COLORS[2], PANEL_COLORS[0]
        else:
            return PANEL_COLORS[1], PANEL_COLORS[0]


def angle_in_sector(theta, start, end):
    """Check if theta is in sector from start to end (counterclockwise)."""
    theta = theta % 360
    start = start % 360
    end = end % 360
    if start <= end:
        return (theta >= start) & (theta < end)
    else:
        return (theta >= start) | (theta < end)


def wrap_angle(angle):
    """Wrap angle to [-180, 180]."""
    return ((angle + 180) % 360) - 180


class AorticSegmentor:
    
    STEP_NAVIGATE = 0
    STEP_ANNULUS = 1
    STEP_STJ = 2
    STEP_ILT_MS = 3  # Central Coaptation / Interleaflet Triangle / Membranous Septum / LVOT
    STEP_GENERATE = 4
    STEP_GENERATE = 5  # Generate Masks
    
    def __init__(self, data_dir: Path, output_dir: Path, totalseg_dir: Optional[Path] = None,
                 csv_file: Optional[Path] = None, id_column: str = 'mrn',
                 exclude_column: Optional[str] = None, exclude_value: Optional[str] = None,
                 reprocess: bool = False, valve_type_filter: Optional[str] = None,
                 poor_only: bool = False, demo: bool = False,
                 case_filter: Optional[str] = None):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.demo = demo
        self.case_filter = case_filter
        
        # Auto-detect NPZ vs DICOM mode
        npz_files = list(data_dir.glob('*.npz'))
        self.npz_mode = len(npz_files) > 0
        if self.npz_mode:
            logger.info(f"NPZ mode: found {len(npz_files)} .npz files in {data_dir}")
        else:
            logger.info(f"DICOM mode: scanning subdirectories in {data_dir}")
        
        # Default totalseg_dir depends on data layout
        if totalseg_dir is not None:
            self.totalseg_dir = totalseg_dir
        elif self.npz_mode:
            # NPZ dir is typically data/raw/npz_deidentified/
            # totalseg_outputs is at data/totalseg_outputs/
            self.totalseg_dir = data_dir.parent.parent / 'totalseg_outputs'
        else:
            self.totalseg_dir = data_dir.parent / 'totalseg_outputs'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_file = csv_file
        self.id_column = id_column
        self.exclude_column = exclude_column
        self.exclude_value = exclude_value
        self.reprocess = reprocess
        self.valve_type_filter = valve_type_filter
        self.poor_only = poor_only
        
        self.cases = self._find_cases()
        self.current_idx = 0
        
        self.volume = None
        self.mpr = None
        self.state = None
        
        # TotalSegmentator mask data
        self.totalseg_mask = None
        self.totalseg_metadata = None  # Contains spacing, origin, crop_info
        self.totalseg_visible = True
        self.totalseg_overlays = [None, None, None]
        
        self.fig = None
        self.ax_panels = []
        self.slice_imgs = [None, None, None]
        
        # Generated masks
        self.lvot_mask = None
        self.cusp_masks = [None, None, None]  # R, L, N
        self.ms_surface_mask = None  # MS surface (1mm thick)
        
        self.mask_overlays = [None, None, None]
        self.ilt_ms_mask_overlays = [None, None, None]
        self.masks_visible = False
        
        # Annulus nadir point placement state
        self.annulus_nadir_idx = 0  # 0=Right, 1=Left, 2=Non
        self.annulus_nadir_temp = []  # Temporary storage during collection
        self.annulus_nadir_artists = []  # Visual markers
        
        # ILT/MS point placement state
        # Phase 0: collecting CC (central coaptation)
        # Phase 1: collecting commissures (0=RN, 1=LR, 2=NL)
        # Phase 2: collecting MS nadir
        # Phase 3: collecting boundary points at 5 levels
        self.ilt_ms_phase = 0
        self.ilt_ms_cc_temp = None  # Temporary CC point
        self.ilt_ms_commissure_idx = 0  # 0=RN, 1=LR, 2=NL
        self.ilt_ms_commissure_temp = []  # Temporary commissure points
        self.ilt_ms_nadir_temp = None  # Temporary MS nadir point
        self.ilt_ms_level = 0  # Current level (0-4)
        self.ilt_ms_level_points = []  # Points at current level
        self.ilt_ms_artists = []  # Visual markers
        
        # LVOT septal boundary point placement state
        # 2 levels: midpoint, LVOT plane
        self.lvot_septal_level = 0
        self.lvot_septal_level_points = []
        self.lvot_septal_artists = []
        
        self.mask_spacing = None
        self.mask_origin = None
        self.overall_mask = None
        
        self.current_step = self.STEP_NAVIGATE
        
        self.dragging = False
        self.drag_type = None
        self.drag_panel = None
        self.drag_start_mouse = None
        self.drag_start_intersection = None
        self.preview_intersection = None
        self.crosshair_offset = np.array([0.0, 0.0])
        
        self.drag_start_frame = None
        self.preview_frame = None
        self.total_rotation_angle = 0.0
        self.last_drag_angle = None
        
        # Plane line dragging state (physical coordinates)
        self.drag_start_center = None
        self.drag_start_normal = None
        
        self.crosshair_artists = [[], [], []]
        self.plane_line_artists = []
        self.current_z_artists = []
        
        self.buttons = {}
        self.button_axes = {}
        self.set_buttons = {}
        self.set_button_axes = {}
        self.reset_buttons = {}
        self.reset_button_axes = {}
        
        # CT Quality buttons
        self.quality_accept_btn = None
        self.quality_poor_btn = None
        self.ax_quality_accept = None
        self.ax_quality_poor = None
        
        # Valve type buttons (6 types now)
        self.valve_btns = {}
        self.valve_btn_axes = {}
        
        # TotalSegmentator toggle button
        self.totalseg_toggle_btn = None
        self.ax_totalseg_toggle = None
        
        # Masks toggle button
        self.masks_toggle_btn = None
        self.ax_masks_toggle = None
        
        # ILT/MS mask toggle button
        self.ilt_ms_mask_toggle_btn = None
        self.ax_ilt_ms_mask_toggle = None
        self.ilt_ms_mask_visible = False
        
        # Boundary spline toggle button
        self.boundary_toggle_btn = None
        self.ax_boundary_toggle = None
        self.boundary_visible = False
        self.boundary_artists = []  # Artists for boundary splines
    
    def _find_cases(self):
        """Find valid cases, optionally filtered by CSV exclusions.
        
        In NPZ mode, cases are .npz files in data_dir.
        In DICOM mode, cases are subdirectories in data_dir.
        """
        import pandas as pd
        
        def extract_patient_id(name):
            """Extract patient ID from case name (handles PatientID_Date format)."""
            parts = name.rsplit('_', 1)
            if len(parts) == 2 and len(parts[1]) == 8 and parts[1].isdigit():
                return parts[0]
            return name
        
        def case_name(path):
            """Get the case name from a Path (stem for NPZ, name for dirs)."""
            return path.stem if self.npz_mode else path.name
        
        # Discover all cases
        if self.npz_mode:
            all_cases = sorted(
                self.data_dir.glob('*.npz'),
                key=lambda p: p.stem
            )
        else:
            all_cases = sorted([
                f for f in self.data_dir.iterdir()
                if f.is_dir() and '_' in f.name
                and not f.name.startswith(('download', 'excluded', '.'))
            ])
        
        # Apply --case filter first (overrides everything else)
        if self.case_filter:
            matched = [c for c in all_cases if case_name(c) == self.case_filter]
            if not matched:
                # Try partial match
                matched = [c for c in all_cases if self.case_filter in case_name(c)]
            if matched:
                logger.info(f"Case filter matched {len(matched)} case(s): {[case_name(c) for c in matched]}")
                return matched
            else:
                logger.error(f"Case filter '{self.case_filter}' matched no cases")
                return []
        
        # CSV filtering
        if self.csv_file is None or self.exclude_column is None or self.exclude_value is None:
            logger.info(f"Found {len(all_cases)} cases")
            return all_cases
        
        if not self.csv_file.exists():
            logger.warning(f"CSV file not found: {self.csv_file}, using all cases")
            return all_cases
        
        df = pd.read_csv(self.csv_file)
        
        if self.id_column not in df.columns:
            logger.warning(f"ID column '{self.id_column}' not in CSV, using all cases")
            return all_cases
        
        if self.exclude_column not in df.columns:
            logger.warning(f"Exclude column '{self.exclude_column}' not in CSV, using all cases")
            return all_cases
        
        n_total_csv = len(df)
        exclude_mask = df[self.exclude_column].apply(
            lambda x: str(x).strip().lower() == self.exclude_value.lower() if pd.notna(x) else False
        )
        n_excluded_csv = exclude_mask.sum()
        df_filtered = df[~exclude_mask]
        
        valid_ids = set()
        for pid in df_filtered[self.id_column]:
            pid_str = str(pid).strip().upper()
            valid_ids.add(pid_str)
        
        filtered_cases = []
        excluded_by_csv = []
        for case in all_cases:
            case_id = extract_patient_id(case_name(case)).upper()
            
            if case_id in valid_ids:
                filtered_cases.append(case)
            else:
                excluded_by_csv.append(case_name(case))
        
        logger.info(f"CSV filtering: {n_total_csv} rows, {n_excluded_csv} excluded by {self.exclude_column}={self.exclude_value}")
        logger.info(f"Cases: {len(all_cases)} total, {len(excluded_by_csv)} excluded, {len(filtered_cases)} remaining")
        
        return filtered_cases
    
    def run(self):
        if not self.cases:
            logger.error("No cases found!")
            return
        self._setup_ui()
        # Find first case needing annotation
        start_idx = 0
        if not self.reprocess:
            for i, case in enumerate(self.cases):
                cn = self._get_case_name(case)
                state_file = self.output_dir / cn / 'metadata.json'
                if not state_file.exists():
                    # No metadata - skip if valve_type or poor_only filter active, otherwise open it
                    if self.valve_type_filter or self.poor_only:
                        continue
                    start_idx = i
                    break
                try:
                    existing_state = CaseState.load(state_file)
                    # Skip if valve_type filter active and doesn't match
                    if self.valve_type_filter and existing_state.valve_type != self.valve_type_filter:
                        continue
                    # Skip if poor_only filter active and not poor
                    if self.poor_only and existing_state.ct_quality != 'poor':
                        continue
                    if existing_state.ct_quality == 'prior_valve':
                        continue
                    if existing_state.ct_quality == 'poor':
                        start_idx = i
                        break
                    if existing_state.ct_quality == 'acceptable':
                        expected_ilt = existing_state.expected_ilt_ms_point_count()
                        has_complete = (
                            existing_state.annulus_set and 
                            existing_state.stj_set and 
                            existing_state.ilt_ms_set and 
                            existing_state.lvot_set and
                            len(existing_state.ilt_ms_points) == expected_ilt and
                            len(existing_state.lvot_septal_boundary_points) == 12
                        )
                        if has_complete:
                            continue
                    start_idx = i
                    break
                except Exception:
                    start_idx = i
                    break
            else:
                # All cases are complete or poor/prior_valve
                logger.info("All cases are complete or marked poor/prior_valve")
                start_idx = len(self.cases) - 1  # Load last case for review
        self._load_case(start_idx)
        plt.show()
    
    def _setup_ui(self):
        self.fig = plt.figure(figsize=(16, 10), facecolor=COLORS['bg'])
        self.fig.canvas.manager.set_window_title('Aortic Root Segmentor')
        
        # 2x2 grid: Sagittal/Coronal on top, Controls/Axial on bottom
        # Leave space at bottom for status bar
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], 
                               height_ratios=[1, 1], wspace=0.04, hspace=0.06,
                               left=0.05, right=0.95, top=0.95, bottom=0.12)
        
        self.ax_panels = []
        # Sagittal top-left, Coronal top-right, Axial bottom-right
        panel_positions = [(0, 0), (0, 1), (1, 1)]
        
        for i, (row, col) in enumerate(panel_positions):
            ax = self.fig.add_subplot(gs[row, col])
            ax.set_facecolor(COLORS['panel_bg'])
            ax.set_title(f'{PANEL_NAMES[i]}', fontsize=10, color=PANEL_COLORS[i], fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(PANEL_COLORS[i])
                spine.set_linewidth(2)
            self.ax_panels.append(ax)
        
        # Controls in bottom-left (under Sagittal)
        self.ax_buttons = self.fig.add_subplot(gs[1, 0])
        self.ax_buttons.set_facecolor(COLORS['bg'])
        self.ax_buttons.axis('off')
        
        self._create_buttons()
        
        self.ax_status = self.fig.add_axes([0.15, 0.01, 0.7, 0.025])
        self.ax_status.set_facecolor(COLORS['bg'])
        self.ax_status.axis('off')
        self.status_text = self.ax_status.text(0.5, 0.5, '', fontsize=9, ha='center', va='center', color=COLORS['text'])
        
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
    
    def _make_button(self, x, y, w, h, label, callback, fontsize=8, color=None):
        """Create a button and return (button, axes)."""
        color = color or COLORS['button']
        ax = self.fig.add_axes([x, y, w, h])
        ax.set_facecolor(color)
        btn = Button(ax, label, color=color, hovercolor=COLORS['button_hover'])
        btn.label.set_color(COLORS['text'])
        btn.label.set_fontsize(fontsize)
        btn.on_clicked(callback)
        return btn, ax
    
    def _make_label(self, x, y, w, h, text):
        """Create a text label axes."""
        ax = self.fig.add_axes([x, y, w, h])
        ax.set_facecolor(COLORS['bg'])
        ax.axis('off')
        ax.text(0.0, 0.5, text, fontsize=7, ha='left', va='center', color=COLORS['text_dim'])
    
    def _create_buttons(self):
        btn_height = 0.036
        btn_width = 0.065
        set_btn_width = 0.028
        reset_btn_width = 0.038
        btn_spacing = 0.042
        
        # Controls panel - left column: Quality and Valve
        ctrl_left_x = 0.08
        ctrl_start_y = 0.42
        
        # Quality label and buttons (stacked vertically)
        self._make_label(ctrl_left_x, ctrl_start_y + 0.015, 0.05, 0.018, 'Quality')
        
        quality_btn_width = 0.050
        quality_btn_y = ctrl_start_y - 0.025
        
        self.quality_accept_btn, self.ax_quality_accept = self._make_button(
            ctrl_left_x, quality_btn_y, quality_btn_width, btn_height,
            'OK', lambda e: self._on_quality_button('acceptable'), fontsize=7, color=COLORS['set_unset'])
        
        self.quality_poor_btn, self.ax_quality_poor = self._make_button(
            ctrl_left_x, quality_btn_y - btn_spacing, quality_btn_width, btn_height,
            'Poor', lambda e: self._on_quality_button('poor'), fontsize=7, color=COLORS['set_unset'])
        
        self.quality_prior_btn, self.ax_quality_prior = self._make_button(
            ctrl_left_x, quality_btn_y - 2 * btn_spacing, quality_btn_width, btn_height,
            'Prior', lambda e: self._on_quality_button('prior_valve'), fontsize=7, color=COLORS['set_unset'])
        
        # Valve section below quality
        valve_y = quality_btn_y - 3 * btn_spacing - 0.01
        self._make_label(ctrl_left_x, valve_y + 0.015, 0.05, 0.018, 'Valve')
        
        valve_btn_width = 0.028
        valve_btn_x = ctrl_left_x
        valve_row1_specs = [('tricuspid', 'Tri'), ('type0', 'T0'), ('type1_lr', 'LR')]
        
        for i, (vtype, label) in enumerate(valve_row1_specs):
            btn, ax = self._make_button(
                valve_btn_x + i * (valve_btn_width + 0.002), valve_y - 0.025,
                valve_btn_width, btn_height, label,
                lambda e, vt=vtype: self._on_valve_type_button(vt), fontsize=6, color=COLORS['set_unset'])
            self.valve_btns[vtype] = btn
            self.valve_btn_axes[vtype] = ax
        
        valve_y2 = valve_y - 0.025 - btn_spacing
        valve_row2_specs = [('type1_rn', 'RN'), ('type1_nl', 'NL'), ('type2', 'T2')]
        
        for i, (vtype, label) in enumerate(valve_row2_specs):
            btn, ax = self._make_button(
                valve_btn_x + i * (valve_btn_width + 0.002), valve_y2,
                valve_btn_width, btn_height, label,
                lambda e, vt=vtype: self._on_valve_type_button(vt), fontsize=6, color=COLORS['set_unset'])
            self.valve_btns[vtype] = btn
            self.valve_btn_axes[vtype] = ax
        
        # Controls panel - right column: workflow buttons (vertically centered)
        ctrl_right_x = 0.30
        set_btn_x = ctrl_right_x + btn_width + 0.004
        reset_btn_x = set_btn_x + set_btn_width + 0.002
        
        workflow_specs = [
            ('mpr', 'Navigate', False),
            ('points', 'Ann/CC/MS/LVOT', True),
            ('stj', 'STJ', True),
            ('generate', 'Gen Masks', False),
            ('save', 'Save', False),
        ]
        
        # Center vertically: 5 buttons with btn_spacing, centered around 0.32
        total_height = (len(workflow_specs) - 1) * btn_spacing
        workflow_start_y = 0.32 + total_height / 2
        
        for i, (name, label, has_set_btn) in enumerate(workflow_specs):
            y = workflow_start_y - i * btn_spacing
            
            fontsize = 6 if name == 'points' else 8
            btn, ax = self._make_button(ctrl_right_x, y, btn_width, btn_height, label,
                                        lambda e, n=name: self._on_button(n), fontsize=fontsize)
            btn.label.set_fontweight('medium')
            self.buttons[name] = btn
            self.button_axes[name] = ax
            
            if has_set_btn:
                set_btn, set_ax = self._make_button(
                    set_btn_x, y, set_btn_width, btn_height, 'Set',
                    lambda e, n=name: self._on_set_button(n), fontsize=7, color=COLORS['set_unset'])
                self.set_buttons[name] = set_btn
                self.set_button_axes[name] = set_ax
                
                reset_btn, reset_ax = self._make_button(
                    reset_btn_x, y, reset_btn_width, btn_height, 'Reset',
                    lambda e, n=name: self._on_reset_button(n), fontsize=6, color=COLORS['set_unset'])
                self.reset_buttons[name] = reset_btn
                self.reset_button_axes[name] = reset_ax
        
        # Bottom bar row 1: Zoom (left), Overlays (middle), Nav (right)
        bottom_row1_y = 0.065
        
        # Zoom on left
        zoom_x = 0.05
        self._make_label(zoom_x, bottom_row1_y + 0.035, 0.04, 0.018, 'Zoom')
        for i, (name, label) in enumerate([('zoom_in', '+'), ('zoom_out', '-')]):
            btn, ax = self._make_button(zoom_x + 0.04 + i * 0.034, bottom_row1_y, 0.032, 0.032, label,
                                        lambda e, n=name: self._on_button(n), fontsize=11)
            btn.label.set_fontweight('bold')
            self.buttons[name] = btn
            self.button_axes[name] = ax
        
        # Overlays in middle
        overlay_x = 0.35
        self._make_label(overlay_x, bottom_row1_y + 0.035, 0.06, 0.018, 'Overlays')
        
        self.totalseg_toggle_btn, self.ax_totalseg_toggle = self._make_button(
            overlay_x + 0.05, bottom_row1_y, 0.045, 0.032, 'TSeg',
            self._on_totalseg_toggle, fontsize=7, color=COLORS['totalseg_toggle_on'])
        
        self.masks_toggle_btn, self.ax_masks_toggle = self._make_button(
            overlay_x + 0.10, bottom_row1_y, 0.045, 0.032, 'Masks',
            self._on_masks_toggle, fontsize=7, color=COLORS['set_unset'])
        
        self.ilt_ms_mask_toggle_btn, self.ax_ilt_ms_mask_toggle = self._make_button(
            overlay_x + 0.15, bottom_row1_y, 0.045, 0.032, 'ILT/MS',
            self._on_ilt_ms_mask_toggle, fontsize=7, color=COLORS['set_unset'])
        
        self.boundary_toggle_btn, self.ax_boundary_toggle = self._make_button(
            overlay_x + 0.20, bottom_row1_y, 0.045, 0.032, 'Bound',
            self._on_boundary_toggle, fontsize=7, color=COLORS['set_unset'])
        
        # Nav on right
        nav_x = 0.82
        self._make_label(nav_x, bottom_row1_y + 0.035, 0.03, 0.018, 'Nav')
        for i, (name, label) in enumerate([('prev', '< Prev'), ('next', 'Next >')]):
            btn, ax = self._make_button(nav_x + 0.03 + i * 0.050, bottom_row1_y, 0.045, 0.032, label,
                                        lambda e, n=name: self._on_button(n), fontsize=7)
            self.buttons[name] = btn
            self.button_axes[name] = ax
    
    def _on_masks_toggle(self, event):
        """Toggle generated masks overlay visibility."""
        self.masks_visible = not self.masks_visible
        color = COLORS['totalseg_toggle_on'] if self.masks_visible else COLORS['set_unset']
        self.ax_masks_toggle.set_facecolor(color)
        self.masks_toggle_btn.color = color
        self._update_all_views()
    
    def _on_ilt_ms_mask_toggle(self, event):
        """Toggle ILT/MS surface mask overlay visibility."""
        self.ilt_ms_mask_visible = not self.ilt_ms_mask_visible
        color = COLORS['totalseg_toggle_on'] if self.ilt_ms_mask_visible else COLORS['set_unset']
        self.ax_ilt_ms_mask_toggle.set_facecolor(color)
        self.ilt_ms_mask_toggle_btn.color = color
        self._update_all_views()
    
    def _on_boundary_toggle(self, event):
        """Toggle boundary spline visibility."""
        self.boundary_visible = not self.boundary_visible
        color = COLORS['totalseg_toggle_on'] if self.boundary_visible else COLORS['set_unset']
        self.ax_boundary_toggle.set_facecolor(color)
        self.boundary_toggle_btn.color = color
        self._update_all_views()
    
    def _on_quality_button(self, quality):
        """Handle CT quality button click."""
        self.state.ct_quality = quality
        self._update_quality_buttons()
        self._update_button_colors()
        
        if quality == 'acceptable':
            self._update_status("CT Quality: Acceptable - proceed with annotations")
        elif quality == 'poor':
            self._save_poor_quality_and_next()
        elif quality == 'prior_valve':
            self._save_poor_quality_and_next()  # Same behavior as poor
    
    def _on_valve_type_button(self, valve_type):
        """Handle valve type button click."""
        self.state.valve_type = valve_type
        self._update_valve_type_buttons()
        
        if valve_type == 'type0':
            self.state.cusp_labels = ['anterior', 'posterior']
            self._update_status("Valve Type: Bicuspid Type 0 (2 cusps: A, P)")
        else:
            self.state.cusp_labels = ['right', 'left', 'non']
            type_names = {'tricuspid': 'Tricuspid', 'type1_lr': 'Bicuspid Type 1 L-R',
                         'type1_rn': 'Bicuspid Type 1 R-N', 'type1_nl': 'Bicuspid Type 1 N-L', 'type2': 'Bicuspid Type 2'}
            self._update_status(f"Valve Type: {type_names.get(valve_type, valve_type)}")
        self._update_all_views()
    
    def _update_quality_buttons(self):
        """Update quality button colors based on state."""
        # Reset all to unset
        self.ax_quality_accept.set_facecolor(COLORS['set_unset'])
        self.quality_accept_btn.color = COLORS['set_unset']
        self.ax_quality_poor.set_facecolor(COLORS['set_unset'])
        self.quality_poor_btn.color = COLORS['set_unset']
        self.ax_quality_prior.set_facecolor(COLORS['set_unset'])
        self.quality_prior_btn.color = COLORS['set_unset']
        
        # Highlight selected
        if self.state.ct_quality == 'acceptable':
            self.ax_quality_accept.set_facecolor(COLORS['acceptable_quality'])
            self.quality_accept_btn.color = COLORS['acceptable_quality']
        elif self.state.ct_quality == 'poor':
            self.ax_quality_poor.set_facecolor(COLORS['poor_quality'])
            self.quality_poor_btn.color = COLORS['poor_quality']
        elif self.state.ct_quality == 'prior_valve':
            self.ax_quality_prior.set_facecolor(COLORS['poor_quality'])
            self.quality_prior_btn.color = COLORS['poor_quality']
    
    def _update_valve_type_buttons(self):
        """Update valve type button colors based on state."""
        for vtype, ax in self.valve_btn_axes.items():
            color = COLORS['set_done'] if self.state.valve_type == vtype else COLORS['set_unset']
            ax.set_facecolor(color)
            self.valve_btns[vtype].color = color
    
    def _on_totalseg_toggle(self, event):
        """Toggle TotalSegmentator overlay visibility."""
        self.totalseg_visible = not self.totalseg_visible
        color = COLORS['totalseg_toggle_on'] if self.totalseg_visible else COLORS['totalseg_toggle_off']
        self.ax_totalseg_toggle.set_facecolor(color)
        self.totalseg_toggle_btn.color = color
        self._update_all_views()
    
    def _update_button_colors(self):
        workflow_order = ['mpr', 'points', 'stj', 'generate', 'save']
        step_to_index = {
            self.STEP_NAVIGATE: 0,
            self.STEP_ANNULUS: 1,  # Points step (annulus phase)
            self.STEP_ILT_MS: 1,   # Points step (ilt_ms phase)
            self.STEP_STJ: 2,
            self.STEP_GENERATE: 3,
        }
        
        current_index = step_to_index.get(self.current_step, 0)
        
        is_set = {
            'points': (self.state.annulus_set and self.state.ilt_ms_set and self.state.lvot_set) if self.state else False,
            'stj': self.state.stj_set if self.state else False,
        }
        
        quality_ok = self.state is not None and not self.state.is_poor_quality()
        
        for name, ax in self.button_axes.items():
            if name in ['prev', 'next', 'zoom_in', 'zoom_out']:
                color = COLORS['button']
            elif name in workflow_order:
                btn_index = workflow_order.index(name)
                if btn_index == current_index:
                    color = COLORS['active']
                elif not quality_ok and name not in ['mpr', 'save']:
                    color = COLORS['set_unset']
                else:
                    color = COLORS['button']
            else:
                color = COLORS['button']
            
            ax.set_facecolor(color)
            if name in self.buttons:
                self.buttons[name].color = color
                self.buttons[name].hovercolor = COLORS['button_hover'] if color == COLORS['button'] else color
        
        for name, ax in self.set_button_axes.items():
            if is_set.get(name, False):
                color = COLORS['set_done']
            else:
                color = COLORS['set_unset']
            ax.set_facecolor(color)
            if name in self.set_buttons:
                self.set_buttons[name].color = color
        
        self.fig.canvas.draw_idle()
    
    def _on_button(self, name):
        if self.state and self.state.is_poor_quality():
            if name in ['points', 'stj', 'generate']:
                self._update_status("CT Quality is Poor - cannot annotate. Click Next to skip.")
                return
        
        if name == 'mpr':
            self.current_step = self.STEP_NAVIGATE
            self._update_all_views()
        elif name == 'points':
            self._set_points_mode()
        elif name == 'stj':
            self._set_stj_mode()
        elif name == 'generate':
            self._generate_masks()
        elif name == 'save':
            self._save_and_next()
        elif name == 'zoom_in':
            self.mpr.fov = max(20, self.mpr.fov * 0.8)
            self.mpr.output_spacing = self.mpr.fov / self.mpr.slice_width
            self._update_all_views()
        elif name == 'zoom_out':
            self.mpr.fov = min(500, self.mpr.fov * 1.25)
            self.mpr.output_spacing = self.mpr.fov / self.mpr.slice_width
            self._update_all_views()
        elif name == 'prev':
            self._prev_case()
        elif name == 'next':
            self._next_case()
        
        self._update_button_colors()
    
    def _on_set_button(self, name):
        if name == 'points':
            if self.state and self.state.annulus_plane and self.state.annulus_nadir_points and \
               len(self.state.ilt_ms_points) == self.state.expected_ilt_ms_point_count() and \
               len(self.state.lvot_septal_boundary_points) == 12:
                self.state.annulus_set = True
                self.state.ilt_ms_set = True
                self.state.lvot_set = True
                self.annulus_nadir_temp = []
                self._update_status("All points set (Annulus/CC/ILT/MS/LVOT)")
            else:
                self._update_status("Complete all points first")
                return
        elif name == 'stj':
            if self.state and self.state.stj_plane:
                # Enforce cranial normal (COORDINATE_CONVENTIONS.md Section 2)
                if self.state.annulus_plane:
                    ann_n = self.state.annulus_plane.get_normal_array()
                    stj_n = self.state.stj_plane.get_normal_array()
                    if np.dot(stj_n, ann_n) < 0:
                        self.state.stj_plane.normal = (-stj_n).tolist()
                self.state.stj_set = True
                center = self.state.stj_plane.get_center_array()
                self._update_status(f"STJ set at [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}]")
        
        self._update_button_colors()
        self._update_all_views()
    
    def _on_reset_button(self, name):
        if name == 'points':
            # Reset all points: annulus + CC/ILT/MS/LVOT
            self.state.annulus_plane = None
            self.state.annulus_nadir_points = None
            self.state.annulus_set = False
            self.annulus_nadir_idx = 0
            self.annulus_nadir_temp = []
            self.state.central_coaptation_point = None
            self.state.commissure_points = None
            self.state.ms_nadir_point = None
            self.state.ilt_ms_points = []
            self.state.ilt_ms_set = False
            self.state.lvot_plane = None
            self.state.lvot_septal_boundary_points = []
            self.state.lvot_set = False
            self.ilt_ms_phase = 0
            self.ilt_ms_cc_temp = None
            self.ilt_ms_commissure_idx = 0
            self.ilt_ms_commissure_temp = []
            self.ilt_ms_nadir_temp = None
            self.ilt_ms_level = 0
            self.ilt_ms_level_points = []
            self.lvot_septal_level = 0
            self.lvot_septal_level_points = []
            self._update_status("All points reset.")
        elif name == 'stj':
            self.state.stj_plane = None
            self.state.stj_set = False
            self._update_status("STJ reset.")
        
        self._update_button_colors()
        self._update_all_views()
    
    def _set_points_mode(self):
        """Start combined points collection: Annulus -> CC -> ILT/MS -> LVOT."""
        self.current_step = self.STEP_ANNULUS
        self.annulus_nadir_idx = 0
        self.annulus_nadir_temp = []
        self._update_all_views()
        self._update_button_colors()
        if self.state.is_type0_bicuspid():
            self._update_status("Click ANTERIOR cusp nadir in axial view (1/3)")
        else:
            self._update_status("Click RIGHT cusp nadir in axial view (1/3)")
    
    def _start_ilt_ms_collection(self):
        """Start CC/ILT/MS/LVOT collection after annulus is complete."""
        self.current_step = self.STEP_ILT_MS
        self.ilt_ms_phase = 0
        self.ilt_ms_cc_temp = None
        self.ilt_ms_commissure_idx = 0
        self.ilt_ms_commissure_temp = []
        self.ilt_ms_nadir_temp = None
        self.ilt_ms_level = 0
        self.ilt_ms_level_points = []
        self.state.ilt_ms_points = []
        self._update_all_views()
        self._update_button_colors()
        self._update_status("Annulus done. Click Central Coaptation point in axial view")
    
    def _add_annulus_nadir_point(self, x_pixel, y_pixel):
        """Add a nadir point during annulus 3-point collection."""
        phys_point = self._screen_to_physical(2, x_pixel, y_pixel)
        self.annulus_nadir_temp.append(phys_point.tolist())
        self.annulus_nadir_idx += 1
        
        if self.state.is_type0_bicuspid():
            click_names = ['ANTERIOR', 'POSTERIOR', 'PLANE']
            click_labels = ['ANTERIOR cusp nadir', 'POSTERIOR cusp nadir', 'annulus plane point']
        else:
            click_names = ['RIGHT', 'LEFT', 'NON']
            click_labels = ['RIGHT cusp nadir', 'LEFT cusp nadir', 'NON cusp nadir']
        
        if self.annulus_nadir_idx < 3:
            self._update_all_views()
            self._update_status(f"Click {click_labels[self.annulus_nadir_idx]} in axial view ({self.annulus_nadir_idx + 1}/3)")
        else:
            # All 3 points collected - compute plane and snap MPR
            points = [np.array(p) for p in self.annulus_nadir_temp]
            
            center = (points[0] + points[1] + points[2]) / 3.0
            v1 = points[1] - points[0]
            v2 = points[2] - points[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            
            # Ensure normal points toward aorta (superior)
            if hasattr(self, '_aorta_centroid') and hasattr(self, '_lv_centroid'):
                lv_to_aorta = self._aorta_centroid - self._lv_centroid
                if np.dot(normal, lv_to_aorta) < 0:
                    normal = -normal
            elif normal[2] < 0:
                normal = -normal
            
            # For type0, only store first 2 points as nadir points
            if self.state.is_type0_bicuspid():
                self.state.annulus_nadir_points = self.annulus_nadir_temp[:2]
            else:
                self.state.annulus_nadir_points = self.annulus_nadir_temp.copy()
            
            self.state.annulus_plane = SegmentationPlane(
                center=center.tolist(),
                normal=normal.tolist()
            )
            
            # Compute frame from plane and snap MPR
            frame = self._compute_frame_from_plane(center, normal)
            self.mpr.frame = frame
            self.mpr.intersection = center.copy()
            
            # Store annulus set and automatically continue to CC/ILT/MS/LVOT
            self.state.annulus_set = True
            self._update_all_views()
            self._update_button_colors()
            
            # Automatically continue to CC/ILT/MS collection
            self._start_ilt_ms_collection()
    
    def _compute_frame_from_plane(self, center, normal):
        """Compute MPR frame from plane center and normal."""
        normal = normal / np.linalg.norm(normal)
        
        # Use anterior direction for consistent orientation
        anterior_dir = np.array([0.0, -1.0, 0.0])
        anterior_in_plane = anterior_dir - np.dot(anterior_dir, normal) * normal
        
        if np.linalg.norm(anterior_in_plane) > 0.1:
            frame_y = anterior_in_plane / np.linalg.norm(anterior_in_plane)
            frame_x = np.cross(frame_y, normal)
            frame_x = frame_x / np.linalg.norm(frame_x)
        else:
            # Fallback
            frame_x = np.array([1.0, 0.0, 0.0])
            frame_x = frame_x - np.dot(frame_x, normal) * normal
            frame_x = frame_x / np.linalg.norm(frame_x)
            frame_y = np.cross(normal, frame_x)
        
        # Ensure right-handed
        if np.dot(np.cross(frame_x, frame_y), normal) < 0:
            frame_x = -frame_x
        
        return np.array([frame_x, frame_y, normal])
    
    def _draw_annulus_nadir_points(self):
        """Draw annulus nadir points on axial panel - only during STEP_ANNULUS."""
        for artist in self.annulus_nadir_artists:
            artist.remove()
        self.annulus_nadir_artists = []
        
        # Only show during annulus step
        if self.current_step != self.STEP_ANNULUS:
            return
        
        # Get points to draw (temp points during collection)
        points = self.annulus_nadir_temp if self.annulus_nadir_temp else None
        
        if not points:
            return
        
        ax = self.ax_panels[2]
        if self.state.is_type0_bicuspid():
            colors = ['red', 'green', 'white']
            labels = ['A', 'P', 'Pl']
        else:
            colors = ['red', 'green', 'blue']
            labels = ['R', 'L', 'N']
        
        # Get axial view normal for distance check
        _, _, normal = self.mpr.get_plane_axes(2)
        
        for i, pt in enumerate(points):
            pt_arr = np.array(pt)
            # Distance from point to current slice along view normal
            dist = abs(np.dot(pt_arr - self.mpr.intersection, normal))
            
            # Only draw if within 1 slice of current view
            if dist > self.mpr.output_spacing:
                continue
            
            px, py = self._physical_to_screen(2, pt_arr)
            marker, = ax.plot(px, py, 'o', color=colors[i], markersize=6,
                             markeredgecolor='white', markeredgewidth=2)
            self.annulus_nadir_artists.append(marker)
            text = ax.text(px + 8, py + 8, labels[i], color=colors[i],
                          fontsize=10, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
            self.annulus_nadir_artists.append(text)
    
    def _set_stj_mode(self):
        if not self.state.annulus_set:
            self._update_status("Set annulus first!")
            return
        
        self.current_step = self.STEP_STJ
        
        # Recenter on annulus plane
        self.mpr.intersection = self.state.annulus_plane.get_center_array().copy()
        
        if self.state.stj_plane is None:
            annulus_center = self.state.annulus_plane.get_center_array()
            annulus_normal = self.state.annulus_plane.get_normal_array()
            stj_center = annulus_center + 15.0 * annulus_normal
            self.state.stj_plane = SegmentationPlane(
                center=stj_center.tolist(),
                normal=annulus_normal.tolist()
            )
        
        self._update_all_views()
        self._update_button_colors()
        self._update_status("Drag STJ line (cyan) to position/angle, then click Set.")
    
    def _add_ilt_ms_point(self, x_pixel, y_pixel):
        """Add a point during CC/ILT/MS/LVOT collection."""
        phys_point = self._screen_to_physical(2, x_pixel, y_pixel)
        
        if self.ilt_ms_phase == 0:
            # Collecting CC point
            self.ilt_ms_cc_temp = phys_point.tolist()
            self.state.central_coaptation_point = self.ilt_ms_cc_temp
            self.ilt_ms_phase = 1
            self._update_all_views()
            if self.state.is_type0_bicuspid():
                self._update_status("Click L commissure in axial view (1/2)")
            else:
                self._update_status("Click RN commissure in axial view (1/3)")
        
        elif self.ilt_ms_phase == 1:
            # Collecting commissure points
            self.ilt_ms_commissure_temp.append(phys_point.tolist())
            self.ilt_ms_commissure_idx += 1
            
            n_comm = self.state.expected_commissure_count()
            if self.state.is_type0_bicuspid():
                commissure_names = ['L', 'R']
            else:
                commissure_names = ['RN', 'LR', 'NL']
            if self.ilt_ms_commissure_idx < n_comm:
                self._update_all_views()
                self._update_status(f"Click {commissure_names[self.ilt_ms_commissure_idx]} commissure in axial view ({self.ilt_ms_commissure_idx + 1}/{n_comm})")
            else:
                # Done with commissures, move to MS nadir
                self.state.commissure_points = self.ilt_ms_commissure_temp.copy()
                self.ilt_ms_phase = 2
                self._update_all_views()
                self._update_status("Click MS nadir (most inferior point of membranous septum)")
        
        elif self.ilt_ms_phase == 2:
            # Collecting MS nadir point
            self.ilt_ms_nadir_temp = phys_point.tolist()
            self.state.ms_nadir_point = self.ilt_ms_nadir_temp
            self.ilt_ms_phase = 3
            self.ilt_ms_level = 0
            self.ilt_ms_level_points = []
            self._move_to_ilt_ms_level(0)
            self._update_all_views()
            if self.state.is_type0_bicuspid():
                self._update_status("Click 4 points for MS boundary at level 1/4 (annulus to MS nadir)")
            else:
                self._update_status("Click 4 points for ILT/MS boundary at level 1/5")
        
        elif self.ilt_ms_phase == 3:
            # Collecting ILT/MS boundary points at levels
            self.ilt_ms_level_points.append(phys_point.tolist())
            
            if len(self.ilt_ms_level_points) == 4:
                # Level complete
                self.state.ilt_ms_points.extend(self.ilt_ms_level_points)
                self.ilt_ms_level += 1
                
                # Type0: 4 levels, Tricuspid: 5 levels
                max_levels = 4 if self.state.is_type0_bicuspid() else 5
                
                if self.ilt_ms_level < max_levels:
                    self.ilt_ms_level_points = []
                    self._move_to_ilt_ms_level(self.ilt_ms_level)
                    self._update_all_views()
                    if self.state.is_type0_bicuspid():
                        self._update_status(f"Click 4 points for MS boundary at level {self.ilt_ms_level + 1}/{max_levels}")
                    else:
                        self._update_status(f"Click 4 points for ILT/MS boundary at level {self.ilt_ms_level + 1}/{max_levels}")
                else:
                    # ILT/MS complete, auto-calculate LVOT plane and continue to LVOT boundary
                    self._setup_lvot_boundary_collection()
            else:
                self._update_all_views()
                max_levels = 4 if self.state.is_type0_bicuspid() else 5
                self._update_status(f"Point {len(self.ilt_ms_level_points)}/4 at level {self.ilt_ms_level + 1}/{max_levels}")
        
        elif self.ilt_ms_phase == 4:
            # Collecting LVOT septal boundary points
            self.lvot_septal_level_points.append(phys_point.tolist())
            
            # Level 0 (MS nadir): 3 clicked points, MS nadir inserted at position 2
            # Levels 1-2: 4 clicked points each
            points_needed = 3 if self.lvot_septal_level == 0 else 4
            
            if len(self.lvot_septal_level_points) == points_needed:
                # Level complete
                if self.lvot_septal_level == 0:
                    # Insert MS nadir at position 1 (so order becomes: pt0, MS_nadir, pt1, pt2)
                    self.lvot_septal_level_points.insert(1, self.state.ms_nadir_point)
                
                self.state.lvot_septal_boundary_points.extend(self.lvot_septal_level_points)
                self.lvot_septal_level += 1
                
                if self.lvot_septal_level < 3:
                    self.lvot_septal_level_points = []
                    self._move_to_lvot_septal_level(self.lvot_septal_level)
                    level_names = ['MS nadir', 'midpoint', 'LVOT plane']
                    self._update_all_views()
                    self._update_status(f"Click 4 LVOT boundary points at {level_names[self.lvot_septal_level]} ({self.lvot_septal_level + 1}/3)")
                else:
                    self._update_all_views()
                    self._update_status("All points complete. Click Set to confirm.")
            else:
                self._update_all_views()
                self._update_status(f"Point {len(self.lvot_septal_level_points)}/{points_needed} at LVOT level {self.lvot_septal_level + 1}/3")
    
    def _setup_lvot_boundary_collection(self):
        """Auto-calculate LVOT plane and start LVOT boundary collection."""
        # Keep original MS nadir point (do not overwrite with level 5 point)
        
        # Auto-calculate LVOT plane: 5mm below MS nadir along annulus normal
        ms_nadir = np.array(self.state.ms_nadir_point)
        annulus_normal = self.state.annulus_plane.get_normal_array()
        annulus_normal = annulus_normal / np.linalg.norm(annulus_normal)
        lvot_center = ms_nadir - 5.0 * annulus_normal  # 5mm below MS nadir (toward LVOT)
        
        self.state.lvot_plane = SegmentationPlane(
            center=lvot_center.tolist(),
            normal=annulus_normal.tolist()
        )
        
        # Setup LVOT boundary collection
        self.state.lvot_septal_boundary_points = []
        self.lvot_septal_level = 0
        self.lvot_septal_level_points = []
        self.ilt_ms_phase = 4
        
        # Move to first level (MS nadir)
        self._move_to_lvot_septal_level(0)
        
        self._update_all_views()
        self._update_status("ILT/MS complete. Click 3 LVOT boundary points at MS nadir (1/3) - order: pt0, [MS nadir], pt1, pt2")
    
    def _move_to_ilt_ms_level(self, level_idx):
        """Move to a specific ILT/MS level along the annulus normal.
        
        Tricuspid: interpolates from RN commissure to MS nadir (5 levels at t=0.2,0.4,0.6,0.8,0.95)
        Type0: interpolates from annulus center to MS nadir (4 levels at t=0.25,0.5,0.75,0.95)
        """
        if self.state.commissure_points is None or self.state.ms_nadir_point is None:
            return
        
        ms_nadir = np.array(self.state.ms_nadir_point)
        
        if self.state.is_type0_bicuspid():
            # Type0: interpolate from annulus plane center to MS nadir (4 levels)
            top_point = self.state.annulus_plane.get_center_array()
            t_values = [0.25, 0.5, 0.75, 0.95]
        else:
            # Tricuspid: interpolate from RN commissure to MS nadir (5 levels)
            top_point = np.array(self.state.commissure_points[0])
            t_values = [0.2, 0.4, 0.6, 0.8, 0.95]
        
        t = t_values[level_idx]
        
        # Interpolate along the annulus normal direction
        target_center = top_point * (1 - t) + ms_nadir * t
        self.mpr.intersection = target_center.copy()
        
        self.ilt_ms_level_points = []
    
    def _draw_ilt_ms_points(self):
        """Draw CC/ILT/MS points on axial panel."""
        for artist in self.ilt_ms_artists:
            artist.remove()
        self.ilt_ms_artists = []
        
        if self.current_step != self.STEP_ILT_MS:
            return
        
        ax = self.ax_panels[2]
        _, _, normal = self.mpr.get_plane_axes(2)
        
        # Draw CC point
        cc_pt = self.ilt_ms_cc_temp if self.ilt_ms_phase == 0 else self.state.central_coaptation_point
        if cc_pt:
            pt_arr = np.array(cc_pt)
            dist = abs(np.dot(pt_arr - self.mpr.intersection, normal))
            if dist <= self.mpr.output_spacing:
                px, py = self._physical_to_screen(2, pt_arr)
                marker, = ax.plot(px, py, 'o', color='white', markersize=6,
                                 markeredgecolor='black', markeredgewidth=2)
                self.ilt_ms_artists.append(marker)
                text = ax.text(px + 8, py + 8, 'CC', color='white',
                              fontsize=10, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
                self.ilt_ms_artists.append(text)
        
        # Draw commissure points
        if self.state.is_type0_bicuspid():
            commissure_colors = ['cyan', 'orange']
            commissure_labels = ['L', 'R']
        else:
            commissure_colors = ['orange', 'cyan', 'magenta']
            commissure_labels = ['RN', 'LR', 'NL']
        points = self.ilt_ms_commissure_temp if self.ilt_ms_phase == 1 else (self.state.commissure_points or [])
        
        for i, pt in enumerate(points):
            pt_arr = np.array(pt)
            dist = abs(np.dot(pt_arr - self.mpr.intersection, normal))
            if dist > self.mpr.output_spacing:
                continue
            px, py = self._physical_to_screen(2, pt_arr)
            marker, = ax.plot(px, py, 'o', color=commissure_colors[i], markersize=6,
                             markeredgecolor='white', markeredgewidth=2)
            self.ilt_ms_artists.append(marker)
            text = ax.text(px + 8, py + 8, commissure_labels[i], color=commissure_colors[i],
                          fontsize=10, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
            self.ilt_ms_artists.append(text)
        
        # Draw MS nadir point (but hide during phase 3 when collecting boundary points)
        if self.ilt_ms_phase != 3:
            nadir_pt = self.ilt_ms_nadir_temp if self.ilt_ms_phase == 2 else self.state.ms_nadir_point
            if nadir_pt:
                pt_arr = np.array(nadir_pt)
                dist = abs(np.dot(pt_arr - self.mpr.intersection, normal))
                if dist <= self.mpr.output_spacing:
                    px, py = self._physical_to_screen(2, pt_arr)
                    marker, = ax.plot(px, py, 's', color='yellow', markersize=6,
                                     markeredgecolor='white', markeredgewidth=2)
                    self.ilt_ms_artists.append(marker)
                    text = ax.text(px + 8, py + 8, 'MS', color='yellow',
                                  fontsize=10, fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
                    self.ilt_ms_artists.append(text)
        
        # Draw current level boundary points
        for pt in self.ilt_ms_level_points:
            pt_arr = np.array(pt)
            dist = abs(np.dot(pt_arr - self.mpr.intersection, normal))
            if dist <= self.mpr.output_spacing:
                px, py = self._physical_to_screen(2, pt_arr)
                marker, = ax.plot(px, py, 'o', color='lime', markersize=5,
                                 markeredgecolor='white', markeredgewidth=2)
                self.ilt_ms_artists.append(marker)
    
    def _move_to_lvot_septal_level(self, level_idx):
        """Move to a specific LVOT septal boundary level."""
        if self.state.ms_nadir_point is None or self.state.lvot_plane is None:
            return
        
        ms_nadir = np.array(self.state.ms_nadir_point)
        lvot_center = self.state.lvot_plane.get_center_array()
        
        # 3 levels: 0 = MS nadir (0.0), 1 = midpoint (0.5), 2 = LVOT plane (1.0)
        t_values = [0.0, 0.5, 1.0]
        t = t_values[level_idx]
        
        target_center = ms_nadir * (1 - t) + lvot_center * t
        self.mpr.intersection = target_center.copy()
        
        self.lvot_septal_level_points = []
    
    def _draw_lvot_septal_points(self):
        """Draw LVOT septal boundary points on axial panel."""
        for artist in self.lvot_septal_artists:
            artist.remove()
        self.lvot_septal_artists = []
        
        # Draw during phase 4 of ILT_MS step
        if self.current_step != self.STEP_ILT_MS or self.ilt_ms_phase != 4:
            return
        
        ax = self.ax_panels[2]
        _, _, normal = self.mpr.get_plane_axes(2)
        
        # Draw current level boundary points
        for pt in self.lvot_septal_level_points:
            pt_arr = np.array(pt)
            dist = abs(np.dot(pt_arr - self.mpr.intersection, normal))
            if dist <= self.mpr.output_spacing:
                px, py = self._physical_to_screen(2, pt_arr)
                marker, = ax.plot(px, py, 'o', color='magenta', markersize=5,
                                 markeredgecolor='white', markeredgewidth=2)
                self.lvot_septal_artists.append(marker)
    
    def _draw_boundary_splines(self):
        """Draw ILT/MS and LVOT septal boundary splines at current z-level on axial panel."""
        # Clear existing boundary artists
        for artist in self.boundary_artists:
            artist.remove()
        self.boundary_artists = []
        
        if not self.boundary_visible:
            return
        
        # Need ILT/MS points and state to draw boundaries
        if not self.state.ilt_ms_set or len(self.state.ilt_ms_points) != self.state.expected_ilt_ms_point_count():
            return
        
        if self.state.annulus_plane is None or self.state.lvot_plane is None:
            return
        
        ax = self.ax_panels[2]
        
        # Get axis parameters
        annulus_center = self.state.annulus_plane.get_center_array()
        annulus_normal = self.state.annulus_plane.get_normal_array()
        annulus_normal = annulus_normal / np.linalg.norm(annulus_normal)
        lvot_center = self.state.lvot_plane.get_center_array()
        
        total_dist = np.dot(lvot_center - annulus_center, annulus_normal)
        if abs(total_dist) < 0.1:
            return
        
        # Current intersection t value
        current_t = np.dot(self.mpr.intersection - annulus_center, annulus_normal) / total_dist
        
        # Get key t values
        ms_nadir = np.array(self.state.ms_nadir_point)
        
        # Type0: use 2nd MS annular point as top reference; Tricuspid: use RN commissure
        if self.state.is_type0_bicuspid():
            top_reference = np.array(self.state.ilt_ms_points[1])
        else:
            top_reference = np.array(self.state.commissure_points[0])
        
        t_top = np.dot(top_reference - annulus_center, annulus_normal) / total_dist
        t_ms_nadir = np.dot(ms_nadir - annulus_center, annulus_normal) / total_dist
        
        # Check if in boundary range
        t_max = 1.0 if len(self.state.lvot_septal_boundary_points) == 12 else t_ms_nadir
        if current_t < t_top - 0.05 or current_t > t_max + 0.05:
            return
        
        # Build surfaces if needed (reuse from state if available)
        ilt_ms_points = np.array(self.state.ilt_ms_points)
        
        # ILT/MS surface control points - tapers to MS nadir (degenerate single point)
        if self.state.is_type0_bicuspid():
            # Type0: 4 levels from annulus to MS nadir
            ilt_t_local = [0.0, 0.25, 0.5, 0.75, 0.95, 1.0]
            ilt_t_global = [t_top + lt * (t_ms_nadir - t_top) for lt in ilt_t_local]
            ilt_control_pts = [
                np.tile(top_reference, (4, 1)),  # Degenerate at annulus center
                ilt_ms_points[0:4],
                ilt_ms_points[4:8],
                ilt_ms_points[8:12],
                ilt_ms_points[12:16],  # Level 4 at t=0.95
                np.tile(ms_nadir, (4, 1)),  # Degenerate at MS nadir
            ]
        else:
            # Tricuspid: 5 levels from RN commissure to MS nadir
            ilt_t_local = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
            ilt_t_global = [t_top + lt * (t_ms_nadir - t_top) for lt in ilt_t_local]
            ilt_control_pts = [
                np.tile(top_reference, (4, 1)),  # Degenerate at RN commissure
                ilt_ms_points[0:4],
                ilt_ms_points[4:8],
                ilt_ms_points[8:12],
                ilt_ms_points[12:16],
                ilt_ms_points[16:20],  # Level 5 at t=0.95
                np.tile(ms_nadir, (4, 1)),  # Degenerate at MS nadir
            ]
        
        ilt_surface = BSplineBoundarySurface(
            ilt_control_pts, ilt_t_global, annulus_center, annulus_normal, total_dist
        )
        
        # LVOT surface if available
        lvot_surface = None
        if len(self.state.lvot_septal_boundary_points) == 12:
            lvot_septal_points = np.array(self.state.lvot_septal_boundary_points)
            lvot_t_local = [0.0, 0.5, 1.0]
            lvot_t_global = [t_ms_nadir + lt * (1.0 - t_ms_nadir) for lt in lvot_t_local]
            lvot_control_pts = [
                lvot_septal_points[0:4],   # MS nadir level (includes MS nadir as middle point)
                lvot_septal_points[4:8],   # Midpoint level
                lvot_septal_points[8:12],  # LVOT plane level
            ]
            lvot_surface = BSplineBoundarySurface(
                lvot_control_pts, lvot_t_global, annulus_center, annulus_normal, total_dist
            )
        
        # Get contour at current t from appropriate surface
        contour_pts = None
        if current_t <= t_ms_nadir:
            contour_pts = ilt_surface.get_contour_at_t(current_t, n_points=50)
        elif lvot_surface is not None:
            contour_pts = lvot_surface.get_contour_at_t(current_t, n_points=50)
        
        if contour_pts is None or len(contour_pts) == 0:
            return
        
        # Convert to screen coordinates
        screen_pts = []
        for pt in contour_pts:
            px, py = self._physical_to_screen(2, pt)
            screen_pts.append([px, py])
        screen_pts = np.array(screen_pts)
        
        # Draw the spline as yellow line
        line, = ax.plot(screen_pts[:, 0], screen_pts[:, 1], 
                       color='yellow', linewidth=2, linestyle='-', alpha=0.9)
        self.boundary_artists.append(line)
        
        # Draw annotation points (ILT/MS boundary points) as yellow markers if on this slice
        _, _, normal = self.mpr.get_plane_axes(2)
        
        # ILT/MS points
        for i, pt in enumerate(self.state.ilt_ms_points):
            pt_arr = np.array(pt)
            dist = abs(np.dot(pt_arr - self.mpr.intersection, normal))
            if dist <= self.mpr.output_spacing * 1.5:
                px, py = self._physical_to_screen(2, pt_arr)
                marker, = ax.plot(px, py, 'o', color='yellow', markersize=5,
                                 markeredgecolor='black', markeredgewidth=1)
                self.boundary_artists.append(marker)
        
        # LVOT points
        for pt in self.state.lvot_septal_boundary_points:
            pt_arr = np.array(pt)
            dist = abs(np.dot(pt_arr - self.mpr.intersection, normal))
            if dist <= self.mpr.output_spacing * 1.5:
                px, py = self._physical_to_screen(2, pt_arr)
                marker, = ax.plot(px, py, 's', color='yellow', markersize=5,
                                 markeredgecolor='black', markeredgewidth=1)
                self.boundary_artists.append(marker)
    
    def _generate_masks(self):
        """Generate LVOT, cusp, and MS surface masks."""
        if not self.state.annulus_set or not self.state.stj_set or not self.state.ilt_ms_set or not self.state.lvot_set:
            self._update_status("Set Annulus, STJ, CC/ILT/MS, and LVOT first!")
            return
        
        self.current_step = self.STEP_GENERATE
        self._update_status("Generating masks...")
        self._update_button_colors()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
        if self.totalseg_mask is None:
            self._update_status("No TotalSegmentator mask available!")
            return
        
        # Get plane parameters
        annulus_center = self.state.annulus_plane.get_center_array()
        annulus_normal = self.state.annulus_plane.get_normal_array()
        annulus_normal = annulus_normal / np.linalg.norm(annulus_normal)
        stj_center = self.state.stj_plane.get_center_array()
        stj_normal = self.state.stj_plane.get_normal_array()
        lvot_center = self.state.lvot_plane.get_center_array()
        lvot_normal = self.state.lvot_plane.get_normal_array()
        
        lv_label = TOTALSEG_LABELS['ventricle_left']
        aorta_label = TOTALSEG_LABELS['aorta']
        
        ts_spacing = np.array(self.totalseg_metadata['spacing'])
        ts_origin = np.array(self.totalseg_metadata['origin'])
        ts_shape = self.totalseg_mask.shape
        
        # Create physical coordinate grids
        ts_zz, ts_yy, ts_xx = np.mgrid[0:ts_shape[0], 0:ts_shape[1], 0:ts_shape[2]]
        phys_x = ts_origin[0] + ts_xx * ts_spacing[0]
        phys_y = ts_origin[1] + ts_yy * ts_spacing[1]
        phys_z = ts_origin[2] + ts_zz * ts_spacing[2]
        phys_coords = np.stack([phys_x, phys_y, phys_z], axis=-1)
        
        def signed_dist_vec(coords, center, normal):
            return np.sum((coords - center) * normal, axis=-1)
        
        # Step 1-3: Compute signed distances and clip masks
        dist_annulus = signed_dist_vec(phys_coords, annulus_center, annulus_normal)
        dist_stj = signed_dist_vec(phys_coords, stj_center, stj_normal)
        dist_lvot = signed_dist_vec(phys_coords, lvot_center, lvot_normal)
        
        lv_mask = (self.totalseg_mask == lv_label)
        aorta_mask = (self.totalseg_mask == aorta_label)
        
        lv_clipped = lv_mask & (dist_lvot > 0)
        aorta_clipped = aorta_mask & (dist_stj < 0)
        between_planes = (dist_lvot > 0) & (dist_stj < 0)
        
        # Step 4: Combine and fill holes
        overall_mask = (lv_clipped | aorta_clipped) & between_planes
        
        struct = ndimage.generate_binary_structure(3, 2)
        dilated = ndimage.binary_dilation(overall_mask, structure=struct, iterations=3)
        dilated = dilated & between_planes
        
        for z_idx in range(ts_shape[0]):
            if np.any(dilated[z_idx]):
                dilated[z_idx] = ndimage.binary_fill_holes(dilated[z_idx])
        
        filled = ndimage.binary_erosion(dilated, structure=struct, iterations=2)
        overall_mask = overall_mask | (filled & between_planes)
        
        # Step 5: Keep only connected component near annulus center
        labeled_mask, num_components = ndimage.label(overall_mask)
        
        if num_components > 1:
            ann_voxel_x = int(round((annulus_center[0] - ts_origin[0]) / ts_spacing[0]))
            ann_voxel_y = int(round((annulus_center[1] - ts_origin[1]) / ts_spacing[1]))
            ann_voxel_z = int(round((annulus_center[2] - ts_origin[2]) / ts_spacing[2]))
            
            ann_voxel_x = np.clip(ann_voxel_x, 0, ts_shape[2] - 1)
            ann_voxel_y = np.clip(ann_voxel_y, 0, ts_shape[1] - 1)
            ann_voxel_z = np.clip(ann_voxel_z, 0, ts_shape[0] - 1)
            
            target_label = labeled_mask[ann_voxel_z, ann_voxel_y, ann_voxel_x]
            
            if target_label == 0:
                found = False
                for radius in range(1, 20):
                    if found:
                        break
                    for dz in range(-radius, radius + 1):
                        if found:
                            break
                        for dy in range(-radius, radius + 1):
                            if found:
                                break
                            for dx in range(-radius, radius + 1):
                                vz = np.clip(ann_voxel_z + dz, 0, ts_shape[0] - 1)
                                vy = np.clip(ann_voxel_y + dy, 0, ts_shape[1] - 1)
                                vx = np.clip(ann_voxel_x + dx, 0, ts_shape[2] - 1)
                                if labeled_mask[vz, vy, vx] > 0:
                                    target_label = labeled_mask[vz, vy, vx]
                                    found = True
                                    break
            
            if target_label > 0:
                overall_mask = (labeled_mask == target_label)
        
        # Step 5b: Expand mask to include annotation points (commissures, nadirs)
        all_annotation_points = []
        if self.state.commissure_points:
            all_annotation_points.extend(self.state.commissure_points)
        if self.state.annulus_nadir_points:
            all_annotation_points.extend(self.state.annulus_nadir_points)
        if self.state.central_coaptation_point:
            all_annotation_points.append(self.state.central_coaptation_point)
        if self.state.ms_nadir_point:
            all_annotation_points.append(self.state.ms_nadir_point)
        
        if all_annotation_points:
            overall_mask = self._expand_mask_to_include_points(overall_mask, all_annotation_points, ts_origin, ts_spacing)
        
        # Step 6: Apply ILT/MS and LVOT septal boundary constraints to overall_mask
        overall_mask = self._apply_ilt_ms_boundary_constraints(overall_mask, ts_shape, ts_origin, ts_spacing, annulus_center, annulus_normal)
        overall_mask = self._apply_lvot_septal_boundary_constraints(overall_mask, ts_shape, ts_origin, ts_spacing, annulus_normal)
        
        # Step 7: Split by annulus plane
        lvot_region = overall_mask & (dist_annulus < 0)
        cusp_region = overall_mask & (dist_annulus > 0)
        
        self.lvot_mask = lvot_region.astype(np.uint8)
        
        # Step 8: Divide cusp region using CC and commissure points
        self.cusp_masks = [np.zeros(ts_shape, dtype=np.uint8) for _ in range(3)]
        
        cc_point = np.array(self.state.central_coaptation_point)
        commissures = [np.array(p) for p in self.state.commissure_points]  # RN, LR, NL or L, R
        
        # Compute angles from CC to each commissure in annulus plane
        def get_angle_in_plane(point, center, normal, ref_vec):
            """Get angle of point relative to center in plane perpendicular to normal."""
            diff = point - center
            # Project onto plane
            diff_in_plane = diff - np.dot(diff, normal) * normal
            # Get perpendicular vector in plane
            perp_vec = np.cross(normal, ref_vec)
            perp_vec = perp_vec / np.linalg.norm(perp_vec)
            # Compute angle
            x = np.dot(diff_in_plane, ref_vec)
            y = np.dot(diff_in_plane, perp_vec)
            return np.arctan2(y, x)
        
        # Create reference vector in annulus plane
        if abs(annulus_normal[2]) < 0.9:
            ref = np.array([0, 0, 1])
        else:
            ref = np.array([1, 0, 0])
        ref_vec = ref - np.dot(ref, annulus_normal) * annulus_normal
        ref_vec = ref_vec / np.linalg.norm(ref_vec)
        
        # Get commissure angles
        comm_angles = [get_angle_in_plane(c, cc_point, annulus_normal, ref_vec) for c in commissures]
        
        cusp_zz, cusp_yy, cusp_xx = np.where(cusp_region)
        
        if len(cusp_zz) > 0:
            cusp_phys_x = ts_origin[0] + cusp_xx * ts_spacing[0]
            cusp_phys_y = ts_origin[1] + cusp_yy * ts_spacing[1]
            cusp_phys_z = ts_origin[2] + cusp_zz * ts_spacing[2]
            
            # Compute angle for each voxel
            dx = cusp_phys_x - cc_point[0]
            dy = cusp_phys_y - cc_point[1]
            dz = cusp_phys_z - cc_point[2]
            
            # Project onto annulus plane
            dot_normal = dx * annulus_normal[0] + dy * annulus_normal[1] + dz * annulus_normal[2]
            in_plane_x = dx - dot_normal * annulus_normal[0]
            in_plane_y = dy - dot_normal * annulus_normal[1]
            in_plane_z = dz - dot_normal * annulus_normal[2]
            
            perp_vec = np.cross(annulus_normal, ref_vec)
            perp_vec = perp_vec / np.linalg.norm(perp_vec)
            
            coord_ref = in_plane_x * ref_vec[0] + in_plane_y * ref_vec[1] + in_plane_z * ref_vec[2]
            coord_perp = in_plane_x * perp_vec[0] + in_plane_y * perp_vec[1] + in_plane_z * perp_vec[2]
            
            voxel_angles = np.arctan2(coord_perp, coord_ref)
            
            def angle_in_sector(angles, start, end):
                """Check if angles are in sector from start to end (counter-clockwise)."""
                angles_norm = (angles - start) % (2 * np.pi)
                span = (end - start) % (2 * np.pi)
                return angles_norm <= span
            
            if self.state.is_type0_bicuspid():
                # Type0: 2 commissures (L, R) -> 2 sectors
                a_l, a_r = comm_angles
                # Anterior cusp: L to R
                sector_ant = angle_in_sector(voxel_angles, a_l, a_r)
                # Posterior cusp: R to L
                sector_post = angle_in_sector(voxel_angles, a_r, a_l)
                
                self.cusp_masks[0][cusp_zz[sector_ant], cusp_yy[sector_ant], cusp_xx[sector_ant]] = 1
                self.cusp_masks[1][cusp_zz[sector_post], cusp_yy[sector_post], cusp_xx[sector_post]] = 1
            else:
                # Tricuspid: 3 commissures (RN, LR, NL) -> 3 sectors
                a_rn, a_lr, a_nl = comm_angles
                
                # Right cusp: RN to LR
                sector_right = angle_in_sector(voxel_angles, a_rn, a_lr)
                # Left cusp: LR to NL
                sector_left = angle_in_sector(voxel_angles, a_lr, a_nl)
                # Non cusp: NL to RN
                sector_non = angle_in_sector(voxel_angles, a_nl, a_rn)
                
                self.cusp_masks[0][cusp_zz[sector_right], cusp_yy[sector_right], cusp_xx[sector_right]] = 1
                self.cusp_masks[1][cusp_zz[sector_left], cusp_yy[sector_left], cusp_xx[sector_left]] = 1
                self.cusp_masks[2][cusp_zz[sector_non], cusp_yy[sector_non], cusp_xx[sector_non]] = 1
        
        # Step 9: Generate MS surface mask (1mm thick)
        self.ms_surface_mask = self._generate_ms_surface_mask(ts_shape, ts_origin, ts_spacing, annulus_normal)
        
        self.overall_mask = overall_mask.astype(np.uint8)
        self.mask_spacing = ts_spacing
        self.mask_origin = ts_origin
        
        self.masks_visible = True
        self.ax_masks_toggle.set_facecolor(COLORS['totalseg_toggle_on'])
        self.masks_toggle_btn.color = COLORS['totalseg_toggle_on']
        
        self._update_all_views()
        lvot_count = np.sum(self.lvot_mask > 0)
        cusp_count = sum(np.sum(m > 0) for m in self.cusp_masks if m is not None)
        ms_count = np.sum(self.ms_surface_mask > 0) if self.ms_surface_mask is not None else 0
        self._update_status(f"Masks generated. LVOT: {lvot_count}, Cusps: {cusp_count}, MS: {ms_count}")
    
    def _apply_ilt_ms_boundary_constraints(self, overall_mask, ts_shape, ts_origin, ts_spacing, annulus_center, annulus_normal):
        """Apply ILT/MS + LVOT septal boundary constraints.
        
        Uses BSplineBoundarySurface for boundary point interpolation.
        Keeps all transition zone logic, mask edge detection, Gaussian smoothing, vertical interpolation.
        
        Tricuspid: surface from RN commissure to MS nadir (5 levels)
        Type0: surface from annulus center to MS nadir (4 levels, no interleaflet triangle)
        """
        if len(self.state.ilt_ms_points) != self.state.expected_ilt_ms_point_count() or len(self.state.lvot_septal_boundary_points) != 12:
            return overall_mask
        
        ms_nadir = np.array(self.state.ms_nadir_point)
        ilt_ms_points = np.array(self.state.ilt_ms_points)
        lvot_septal_points = np.array(self.state.lvot_septal_boundary_points)
        lvot_center = self.state.lvot_plane.get_center_array()
        
        # Type0: use 2nd MS annular point as top reference; Tricuspid: use RN commissure
        if self.state.is_type0_bicuspid():
            top_reference = np.array(self.state.ilt_ms_points[1])
        else:
            top_reference = np.array(self.state.commissure_points[0])
        
        total_dist = np.dot(lvot_center - annulus_center, annulus_normal)
        if abs(total_dist) < 0.1:
            return overall_mask
        
        new_mask = overall_mask.copy()
        
        # 2D plane axes
        if abs(annulus_normal[2]) < 0.9:
            ref = np.array([0, 0, 1])
        else:
            ref = np.array([1, 0, 0])
        plane_x = np.cross(annulus_normal, ref)
        plane_x = plane_x / np.linalg.norm(plane_x)
        plane_y = np.cross(annulus_normal, plane_x)
        plane_y = plane_y / np.linalg.norm(plane_y)
        
        # Calculate t values for key points
        t_top = np.dot(top_reference - annulus_center, annulus_normal) / total_dist
        t_ms_nadir = np.dot(ms_nadir - annulus_center, annulus_normal) / total_dist
        
        # Build BSplineBoundarySurface objects for consistent interpolation
        if self.state.is_type0_bicuspid():
            # Type0: 4 levels from annulus to MS nadir (no interleaflet triangle)
            ilt_t_local = [0.0, 0.25, 0.5, 0.75, 0.95, 1.0]
            ilt_t_global = [t_top + lt * (t_ms_nadir - t_top) for lt in ilt_t_local]
            ilt_control_pts = [
                np.tile(top_reference, (4, 1)),  # Degenerate at annulus center
                ilt_ms_points[0:4],
                ilt_ms_points[4:8],
                ilt_ms_points[8:12],
                ilt_ms_points[12:16],  # Level 4 at t=0.95
                np.tile(ms_nadir, (4, 1)),  # Degenerate at MS nadir
            ]
        else:
            # Tricuspid: 5 levels from RN commissure to MS nadir
            ilt_t_local = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
            ilt_t_global = [t_top + lt * (t_ms_nadir - t_top) for lt in ilt_t_local]
            ilt_control_pts = [
                np.tile(top_reference, (4, 1)),  # Degenerate at RN commissure
                ilt_ms_points[0:4],
                ilt_ms_points[4:8],
                ilt_ms_points[8:12],
                ilt_ms_points[12:16],
                ilt_ms_points[16:20],  # Level 5 at t=0.95
                np.tile(ms_nadir, (4, 1)),  # Degenerate at MS nadir
            ]
        
        lvot_t_local = [0.0, 0.5, 1.0]
        lvot_t_global = [t_ms_nadir + lt * (1.0 - t_ms_nadir) for lt in lvot_t_local]
        lvot_control_pts = [
            lvot_septal_points[0:4],   # MS nadir level (includes MS nadir as middle point)
            lvot_septal_points[4:8],   # Midpoint level
            lvot_septal_points[8:12],  # LVOT plane level
        ]
        
        ilt_surface = BSplineBoundarySurface(
            ilt_control_pts, ilt_t_global, annulus_center, annulus_normal, total_dist
        )
        lvot_surface = BSplineBoundarySurface(
            lvot_control_pts, lvot_t_global, annulus_center, annulus_normal, total_dist
        )
        
        # Store surface params for saving
        self.state.ilt_ms_surface_params = ilt_surface.to_dict()
        self.state.lvot_surface_params = lvot_surface.to_dict()
        
        def get_axis_point(t):
            return annulus_center + t * total_dist * annulus_normal
        
        def get_boundary_points_at_t(t):
            """Get boundary points at parameter t using BSplineBoundarySurface.
            
            Returns (points_2d, n_pts) where points are already in 2D plane coordinates.
            At MS nadir level, combines both surfaces for 8 points.
            """
            axis_pt = get_axis_point(t)
            
            if t <= t_top:
                # Degenerate point
                d = top_reference - axis_pt
                pt_2d = np.array([np.dot(d, plane_x), np.dot(d, plane_y)])
                return np.tile(pt_2d, (4, 1)), 4
            
            # At MS nadir, combine both surfaces
            if abs(t - t_ms_nadir) < 0.001:
                ilt_contour = ilt_surface.get_contour_at_t(t, n_points=50)
                lvot_contour = lvot_surface.get_contour_at_t(t, n_points=50)
                
                # Project to 2D
                ilt_2d = []
                for pt in ilt_contour:
                    d = pt - axis_pt
                    ilt_2d.append([np.dot(d, plane_x), np.dot(d, plane_y)])
                
                lvot_2d = []
                for pt in lvot_contour:
                    d = pt - axis_pt
                    lvot_2d.append([np.dot(d, plane_x), np.dot(d, plane_y)])
                
                # Return combined (already dense from contour)
                combined = np.vstack([np.array(ilt_2d), np.array(lvot_2d)])
                return combined, len(combined)
            
            # Use appropriate surface
            if t < t_ms_nadir:
                contour = ilt_surface.get_contour_at_t(t, n_points=50)
            else:
                contour = lvot_surface.get_contour_at_t(t, n_points=50)
            
            # Project to 2D
            pts_2d = []
            for pt in contour:
                d = pt - axis_pt
                pts_2d.append([np.dot(d, plane_x), np.dot(d, plane_y)])
            
            return np.array(pts_2d), len(pts_2d)
        
        # Precompute spline data for discrete t levels
        n_t_levels = 40
        t_values_precompute = np.linspace(t_top, 1.0, n_t_levels)
        
        raw_data = []
        
        for ti, t in enumerate(t_values_precompute):
            axis_point = get_axis_point(t)
            pts_2d, n_pts = get_boundary_points_at_t(t)
            
            # Compute spline endpoint angles
            start_angle = np.arctan2(pts_2d[0, 1], pts_2d[0, 0])
            end_angle = np.arctan2(pts_2d[-1, 1], pts_2d[-1, 0])
            
            # Convert spline points to angle->radius lookup
            spline_angles = np.arctan2(pts_2d[:, 1], pts_2d[:, 0])
            spline_radii = np.sqrt(pts_2d[:, 0]**2 + pts_2d[:, 1]**2)
            sort_idx = np.argsort(spline_angles)
            
            raw_data.append({
                'axis_point': axis_point,
                'start_angle': start_angle,
                'end_angle': end_angle,
                'spline_angles': spline_angles[sort_idx],
                'spline_radii': spline_radii[sort_idx],
            })
        
        # Build boundary_data
        boundary_data = []
        for i, t in enumerate(t_values_precompute):
            boundary_data.append({
                't': t,
                'axis_point': raw_data[i]['axis_point'],
                'spline_angles': raw_data[i]['spline_angles'],
                'spline_radii': raw_data[i]['spline_radii'],
                'start_angle': raw_data[i]['start_angle'],
                'end_angle': raw_data[i]['end_angle'],
            })
        
        # Get voxel bounds
        all_pts = np.vstack([
            top_reference.reshape(1, 3), ilt_ms_points, ms_nadir.reshape(1, 3),
            lvot_septal_points, lvot_center.reshape(1, 3)
        ])
        pts_min = all_pts.min(axis=0) - 5
        pts_max = all_pts.max(axis=0) + 5
        
        vox_min = np.floor((pts_min - ts_origin) / ts_spacing).astype(int)
        vox_max = np.ceil((pts_max - ts_origin) / ts_spacing).astype(int)
        vox_min = np.maximum(vox_min, 0)
        vox_max = np.minimum(vox_max, np.array([ts_shape[2], ts_shape[1], ts_shape[0]]))
        
        vx_range = np.arange(vox_min[0], vox_max[0])
        vy_range = np.arange(vox_min[1], vox_max[1])
        vz_range = np.arange(vox_min[2], vox_max[2])
        
        if len(vx_range) == 0 or len(vy_range) == 0 or len(vz_range) == 0:
            return overall_mask
        
        vx_grid, vy_grid, vz_grid = np.meshgrid(vx_range, vy_range, vz_range, indexing='ij')
        vx_flat = vx_grid.ravel()
        vy_flat = vy_grid.ravel()
        vz_flat = vz_grid.ravel()
        
        px = ts_origin[0] + vx_flat * ts_spacing[0]
        py = ts_origin[1] + vy_flat * ts_spacing[1]
        pz = ts_origin[2] + vz_flat * ts_spacing[2]
        
        dx = px - annulus_center[0]
        dy = py - annulus_center[1]
        dz = pz - annulus_center[2]
        t_all = (dx * annulus_normal[0] + dy * annulus_normal[1] + dz * annulus_normal[2]) / total_dist
        
        valid_t = (t_all >= t_top) & (t_all <= 1.0)
        
        t_valid = t_all[valid_t]
        idx_float = (t_valid - t_top) / (1.0 - t_top) * (n_t_levels - 1)
        idx_lo = np.floor(idx_float).astype(int)
        idx_hi = np.ceil(idx_float).astype(int)
        idx_lo = np.clip(idx_lo, 0, n_t_levels - 1)
        idx_hi = np.clip(idx_hi, 0, n_t_levels - 1)
        frac = idx_float - idx_lo
        frac = np.clip(frac, 0, 1)
        
        vx_valid = vx_flat[valid_t]
        vy_valid = vy_flat[valid_t]
        vz_valid = vz_flat[valid_t]
        px_valid = px[valid_t]
        py_valid = py[valid_t]
        pz_valid = pz[valid_t]
        
        # Process all voxels with interpolation between levels
        # Track spline-protected region for smoothing step
        spline_protected = np.zeros_like(new_mask, dtype=bool)
        
        unique_pairs = set(zip(idx_lo, idx_hi))
        
        for (lo, hi) in unique_pairs:
            pair_mask = (idx_lo == lo) & (idx_hi == hi)
            if not np.any(pair_mask):
                continue
            
            bd_lo = boundary_data[lo]
            bd_hi = boundary_data[hi]
            
            axis_pt = bd_lo['axis_point']
            
            # Use actual spline angular range
            a_min = max(bd_lo['start_angle'], bd_hi['start_angle'])
            a_max = min(bd_lo['end_angle'], bd_hi['end_angle'])
            span = (a_max - a_min) % (2 * np.pi)
            
            lvx = vx_valid[pair_mask]
            lvy = vy_valid[pair_mask]
            lvz = vz_valid[pair_mask]
            lpx = px_valid[pair_mask]
            lpy = py_valid[pair_mask]
            lpz = pz_valid[pair_mask]
            lfrac = frac[pair_mask]
            
            d_x = lpx - axis_pt[0]
            d_y = lpy - axis_pt[1]
            d_z = lpz - axis_pt[2]
            
            pt_2d_x = d_x * plane_x[0] + d_y * plane_x[1] + d_z * plane_x[2]
            pt_2d_y = d_x * plane_y[0] + d_y * plane_y[1] + d_z * plane_y[2]
            
            angles = np.arctan2(pt_2d_y, pt_2d_x)
            radii = np.sqrt(pt_2d_x**2 + pt_2d_y**2)
            
            # Check if voxel is within spline angular range
            angle_norm = (angles - a_min) % (2 * np.pi)
            in_spline_range = angle_norm <= span
            
            # Compute boundary radius from spline
            boundary_r_lo = np.interp(angles, bd_lo['spline_angles'], bd_lo['spline_radii'])
            boundary_r_hi = np.interp(angles, bd_hi['spline_angles'], bd_hi['spline_radii'])
            boundary_r = boundary_r_lo * (1 - lfrac) + boundary_r_hi * lfrac
            
            # For voxels IN spline range: apply full constraint
            should_be_in = (radii <= boundary_r) & in_spline_range
            should_be_out = (radii > boundary_r) & in_spline_range
            
            new_mask[lvz[should_be_in], lvy[should_be_in], lvx[should_be_in]] = True
            new_mask[lvz[should_be_out], lvy[should_be_out], lvx[should_be_out]] = False
            
            # Track spline-protected voxels
            spline_protected[lvz[in_spline_range], lvy[in_spline_range], lvx[in_spline_range]] = True
        
        # Save spline-protected region state before smoothing
        protected_state = new_mask[spline_protected]
        
        # Apply median filter (smooths jagged edges)
        new_mask = ndimage.median_filter(new_mask.astype(np.uint8), size=5).astype(bool)
        
        # Apply Gaussian blur + threshold (additional smoothing)
        blurred = ndimage.gaussian_filter(new_mask.astype(np.float32), sigma=1.0)
        new_mask = blurred > 0.5
        
        # Restore spline-protected region
        new_mask[spline_protected] = protected_state
        
        return ndimage.binary_fill_holes(new_mask)
    
    def _apply_surface_constraint(self, mask, surface, plane_x, plane_y, t_vals, vx, vy, vz, px, py, pz):
        """DEPRECATED - kept for compatibility but not used."""
        pass
    def _expand_mask_to_include_points(self, mask, points, ts_origin, ts_spacing, annulus_center=None, annulus_normal=None):
        """Expand mask to include annotation points that may be outside.
        
        For each point outside the mask:
        1. Find point's 2D position (angle, radius) in annulus plane
        2. Find mask edge radius at that angle
        3. Fill wedge from mask edge to point
        4. Vertical smoothing over distance proportional to gap
        """
        if points is None or len(points) == 0:
            return mask
        
        # Get annulus info if not provided
        if annulus_center is None or annulus_normal is None:
            if self.state.annulus_plane is None:
                return mask
            annulus_center = self.state.annulus_plane.get_center_array()
            annulus_normal = self.state.annulus_plane.get_normal_array()
            annulus_normal = annulus_normal / np.linalg.norm(annulus_normal)
        
        # Build 2D coordinate system
        if abs(annulus_normal[2]) < 0.9:
            ref = np.array([0, 0, 1])
        else:
            ref = np.array([1, 0, 0])
        plane_x = np.cross(annulus_normal, ref)
        plane_x = plane_x / np.linalg.norm(plane_x)
        plane_y = np.cross(annulus_normal, plane_x)
        
        new_mask = mask.copy()
        
        for pt in points:
            pt = np.array(pt)
            
            # Convert to voxel coordinates
            vx = int(round((pt[0] - ts_origin[0]) / ts_spacing[0]))
            vy = int(round((pt[1] - ts_origin[1]) / ts_spacing[1]))
            vz = int(round((pt[2] - ts_origin[2]) / ts_spacing[2]))
            
            # Check bounds
            if not (0 <= vx < mask.shape[2] and 0 <= vy < mask.shape[1] and 0 <= vz < mask.shape[0]):
                continue
            
            # If point already in mask, skip
            if mask[vz, vy, vx]:
                continue
            
            # Project point to 2D in annulus plane
            d_pt = pt - annulus_center
            pt_2d_x = np.dot(d_pt, plane_x)
            pt_2d_y = np.dot(d_pt, plane_y)
            pt_angle = np.arctan2(pt_2d_y, pt_2d_x)
            pt_radius = np.sqrt(pt_2d_x**2 + pt_2d_y**2)
            
            # Find mask edge radius at this angle on this slice
            # Sample mask voxels on this z-slice
            slice_mask = mask[vz, :, :]
            if not np.any(slice_mask):
                continue
            
            # Get mask voxel positions for this slice
            my, mx = np.where(slice_mask)
            if len(mx) == 0:
                continue
            
            mask_px = ts_origin[0] + mx * ts_spacing[0]
            mask_py = ts_origin[1] + my * ts_spacing[1]
            mask_pz = ts_origin[2] + vz * ts_spacing[2]
            
            # Project mask voxels to 2D
            mask_dx = mask_px - annulus_center[0]
            mask_dy = mask_py - annulus_center[1]
            mask_dz = mask_pz - annulus_center[2]
            
            mask_2d_x = mask_dx * plane_x[0] + mask_dy * plane_x[1] + mask_dz * plane_x[2]
            mask_2d_y = mask_dx * plane_y[0] + mask_dy * plane_y[1] + mask_dz * plane_y[2]
            mask_angles = np.arctan2(mask_2d_y, mask_2d_x)
            mask_radii = np.sqrt(mask_2d_x**2 + mask_2d_y**2)
            
            # Find mask edge radius near point's angle (within ~15 degrees)
            angle_diff = np.abs((mask_angles - pt_angle + np.pi) % (2 * np.pi) - np.pi)
            near_angle = angle_diff < 0.26  # ~15 degrees
            
            if not np.any(near_angle):
                # No mask voxels near this angle, use simple expansion
                edge_radius = pt_radius * 0.8
            else:
                edge_radius = mask_radii[near_angle].max()
            
            # Gap distance
            gap = pt_radius - edge_radius
            if gap <= 0:
                continue  # Point is inside mask edge, shouldn't happen
            
            # Vertical smoothing distance proportional to gap (1.5x gap, min 2mm, max 10mm)
            vertical_dist_mm = np.clip(gap * 1.5, 2.0, 10.0)
            vertical_dist_voxels = int(np.ceil(vertical_dist_mm / ts_spacing[2]))
            
            # Angular width for wedge (proportional to gap, ~5-15 degrees)
            angular_width = np.clip(gap / pt_radius * 2, 0.09, 0.26)  # ~5-15 degrees
            
            # Fill wedge on multiple z-slices with interpolated radius
            for dz in range(-vertical_dist_voxels, vertical_dist_voxels + 1):
                sz = vz + dz
                if not (0 <= sz < mask.shape[0]):
                    continue
                
                # Interpolate target radius: full at point's z, back to edge_radius at ends
                frac = 1.0 - abs(dz) / (vertical_dist_voxels + 0.001)
                target_radius = edge_radius + gap * frac
                
                # Fill voxels in wedge from edge_radius to target_radius
                for sy in range(max(0, vy - 20), min(mask.shape[1], vy + 21)):
                    for sx in range(max(0, vx - 20), min(mask.shape[2], vx + 21)):
                        px = ts_origin[0] + sx * ts_spacing[0]
                        py = ts_origin[1] + sy * ts_spacing[1]
                        pz = ts_origin[2] + sz * ts_spacing[2]
                        
                        # Project to 2D
                        dx = px - annulus_center[0]
                        dy = py - annulus_center[1]
                        dz_pt = pz - annulus_center[2]
                        
                        vox_2d_x = dx * plane_x[0] + dy * plane_x[1] + dz_pt * plane_x[2]
                        vox_2d_y = dx * plane_y[0] + dy * plane_y[1] + dz_pt * plane_y[2]
                        vox_angle = np.arctan2(vox_2d_y, vox_2d_x)
                        vox_radius = np.sqrt(vox_2d_x**2 + vox_2d_y**2)
                        
                        # Check if in wedge (angle near point's angle, radius between edge and target)
                        angle_diff = abs((vox_angle - pt_angle + np.pi) % (2 * np.pi) - np.pi)
                        if angle_diff <= angular_width and edge_radius <= vox_radius <= target_radius:
                            new_mask[sz, sy, sx] = True
        
        return new_mask
    
    def _apply_lvot_septal_boundary_constraints(self, overall_mask, ts_shape, ts_origin, ts_spacing, annulus_normal):
        """Placeholder - all work done in _apply_ilt_ms_boundary_constraints now."""
        return overall_mask
    
    def _generate_ms_surface_mask(self, ts_shape, ts_origin, ts_spacing, annulus_normal):
        """Generate MS surface mask as 1mm thick (0.5mm each side) curved surface.
        
        Uses the ILT/MS BSplineBoundarySurface for consistent interpolation.
        
        Tricuspid: surface from RN commissure to MS nadir (5 levels)
        Type0: surface from annulus center to MS nadir (4 levels)
        """
        if len(self.state.ilt_ms_points) != self.state.expected_ilt_ms_point_count():
            return None
        
        ms_mask = np.zeros(ts_shape, dtype=np.uint8)
        
        ms_nadir = np.array(self.state.ms_nadir_point)
        ilt_ms_points = np.array(self.state.ilt_ms_points)
        annulus_center = self.state.annulus_plane.get_center_array()
        lvot_center = self.state.lvot_plane.get_center_array()
        
        # Type0: use 2nd MS annular point as top reference; Tricuspid: use RN commissure
        if self.state.is_type0_bicuspid():
            top_reference = np.array(self.state.ilt_ms_points[1])
        else:
            top_reference = np.array(self.state.commissure_points[0])
        
        # Surface thickness: 0.75mm on each side (1.5mm total)
        thickness = 0.75
        
        # Calculate t values
        total_dist = np.dot(lvot_center - annulus_center, annulus_normal)
        if abs(total_dist) < 0.1:
            return None
        
        t_top = np.dot(top_reference - annulus_center, annulus_normal) / total_dist
        t_ms_nadir = np.dot(ms_nadir - annulus_center, annulus_normal) / total_dist
        
        # Build ILT/MS surface - tapers to MS nadir (degenerate single point)
        if self.state.is_type0_bicuspid():
            # Type0: 4 levels from annulus to MS nadir
            ilt_t_local = [0.0, 0.25, 0.5, 0.75, 0.95, 1.0]
            ilt_t_global = [t_top + lt * (t_ms_nadir - t_top) for lt in ilt_t_local]
            ilt_control_pts = [
                np.tile(top_reference, (4, 1)),  # Degenerate at annulus center
                ilt_ms_points[0:4],
                ilt_ms_points[4:8],
                ilt_ms_points[8:12],
                ilt_ms_points[12:16],  # Level 4 at t=0.95
                np.tile(ms_nadir, (4, 1)),  # Degenerate at MS nadir
            ]
        else:
            # Tricuspid: 5 levels from RN commissure to MS nadir
            ilt_t_local = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
            ilt_t_global = [t_top + lt * (t_ms_nadir - t_top) for lt in ilt_t_local]
            ilt_control_pts = [
                np.tile(top_reference, (4, 1)),  # Degenerate at RN commissure
                ilt_ms_points[0:4],
                ilt_ms_points[4:8],
                ilt_ms_points[8:12],
                ilt_ms_points[12:16],
                ilt_ms_points[16:20],  # Level 5 at t=0.95
                np.tile(ms_nadir, (4, 1)),  # Degenerate at MS nadir
            ]
        
        ilt_surface = BSplineBoundarySurface(
            ilt_control_pts, ilt_t_global, annulus_center, annulus_normal, total_dist
        )
        
        plane_x = ilt_surface.plane_x
        plane_y = ilt_surface.plane_y
        
        # Get bounding box
        all_points = np.vstack([top_reference.reshape(1, 3), ilt_ms_points, ms_nadir.reshape(1, 3)])
        pt_min = all_points.min(axis=0) - thickness - 2
        pt_max = all_points.max(axis=0) + thickness + 2
        
        vox_min = np.floor((pt_min - ts_origin) / ts_spacing).astype(int)
        vox_max = np.ceil((pt_max - ts_origin) / ts_spacing).astype(int)
        vox_min = np.maximum(vox_min, 0)
        vox_max = np.minimum(vox_max, np.array([ts_shape[2], ts_shape[1], ts_shape[0]]))
        
        # Dense t-sampling for smooth surface
        n_t_samples = 80
        t_samples = np.linspace(t_top, t_ms_nadir, n_t_samples)
        contour_data = []
        
        for t in t_samples:
            contour_pts = ilt_surface.get_contour_at_t(t, n_points=100)
            axis_pt = ilt_surface._get_axis_point(t)
            
            # Project to 2D
            contour_2d = []
            for pt in contour_pts:
                d = pt - axis_pt
                contour_2d.append([np.dot(d, plane_x), np.dot(d, plane_y)])
            contour_2d = np.array(contour_2d)
            
            contour_data.append({
                't': t,
                'axis_pt': axis_pt,
                'contour_2d': contour_2d,
            })
        
        # Vectorized processing
        vx_range = np.arange(vox_min[0], vox_max[0])
        vy_range = np.arange(vox_min[1], vox_max[1])
        vz_range = np.arange(vox_min[2], vox_max[2])
        
        if len(vx_range) == 0 or len(vy_range) == 0 or len(vz_range) == 0:
            return ms_mask
        
        vx_grid, vy_grid, vz_grid = np.meshgrid(vx_range, vy_range, vz_range, indexing='ij')
        vx_flat = vx_grid.ravel()
        vy_flat = vy_grid.ravel()
        vz_flat = vz_grid.ravel()
        
        px = ts_origin[0] + vx_flat * ts_spacing[0]
        py = ts_origin[1] + vy_flat * ts_spacing[1]
        pz = ts_origin[2] + vz_flat * ts_spacing[2]
        
        # Compute t for all voxels
        dx = px - annulus_center[0]
        dy = py - annulus_center[1]
        dz = pz - annulus_center[2]
        t_all = (dx * annulus_normal[0] + dy * annulus_normal[1] + dz * annulus_normal[2]) / total_dist
        
        # Filter to ILT/MS region
        valid = (t_all >= t_top) & (t_all <= t_ms_nadir)
        
        t_valid = t_all[valid]
        vx_valid = vx_flat[valid]
        vy_valid = vy_flat[valid]
        vz_valid = vz_flat[valid]
        px_valid = px[valid]
        py_valid = py[valid]
        pz_valid = pz[valid]
        
        # Map to contour level indices
        idx_float = (t_valid - t_top) / (t_ms_nadir - t_top + 1e-10) * (n_t_samples - 1)
        idx_int = np.clip(np.round(idx_float).astype(int), 0, n_t_samples - 1)
        
        # Process by level
        for level_idx in range(n_t_samples):
            level_mask = (idx_int == level_idx)
            if not np.any(level_mask):
                continue
            
            cd = contour_data[level_idx]
            axis_pt = cd['axis_pt']
            contour_2d = cd['contour_2d']
            
            lvx = vx_valid[level_mask]
            lvy = vy_valid[level_mask]
            lvz = vz_valid[level_mask]
            lpx = px_valid[level_mask]
            lpy = py_valid[level_mask]
            lpz = pz_valid[level_mask]
            
            # Project voxels to 2D
            d_x = lpx - axis_pt[0]
            d_y = lpy - axis_pt[1]
            d_z = lpz - axis_pt[2]
            
            pt_2d_x = d_x * plane_x[0] + d_y * plane_x[1] + d_z * plane_x[2]
            pt_2d_y = d_x * plane_y[0] + d_y * plane_y[1] + d_z * plane_y[2]
            
            # Compute distance to contour for each voxel
            for i in range(len(lvx)):
                pt_2d = np.array([pt_2d_x[i], pt_2d_y[i]])
                dists = np.linalg.norm(contour_2d - pt_2d, axis=1)
                min_dist = np.min(dists)
                
                if min_dist <= thickness:
                    ms_mask[lvz[i], lvy[i], lvx[i]] = 1
        
        # Smooth mask: median filter then Gaussian blur + threshold
        if np.sum(ms_mask) > 0:
            ms_mask = ndimage.median_filter(ms_mask, size=3)
            ms_mask_smooth = ndimage.gaussian_filter(ms_mask.astype(np.float32), sigma=1.2)
            ms_mask = (ms_mask_smooth >= 0.5).astype(np.uint8)
        
        return ms_mask
    
    def _screen_to_physical(self, panel_idx, px, py):
        """Convert screen pixel coordinates to physical DICOM coordinates."""
        sw, sh = self.mpr.slice_width, self.mpr.slice_height
        dx_px = px - sw / 2
        dy_px = py - sh / 2
        
        x_axis, y_axis, _ = self.mpr.get_plane_axes(panel_idx)
        
        offset = (dx_px * self.mpr.output_spacing * x_axis + 
                 dy_px * self.mpr.output_spacing * y_axis)
        
        return self.mpr.intersection + offset
    
    def _physical_to_screen(self, panel_idx, phys_point):
        """Convert physical DICOM coordinates to screen pixel coordinates."""
        x_axis, y_axis, _ = self.mpr.get_plane_axes(panel_idx)
        
        diff = phys_point - self.mpr.intersection
        
        dx_mm = np.dot(diff, x_axis)
        dy_mm = np.dot(diff, y_axis)
        
        sw, sh = self.mpr.slice_width, self.mpr.slice_height
        px = sw / 2 + dx_mm / self.mpr.output_spacing
        py = sh / 2 + dy_mm / self.mpr.output_spacing
        
        return px, py
    
    def _load_totalseg_mask(self, case_name: str):
        """Load TotalSegmentator mask for the current case."""
        self.totalseg_mask = None
        self.totalseg_metadata = None
        
        if not self.totalseg_dir.exists():
            logger.info(f"TotalSegmentator directory not found: {self.totalseg_dir}")
            return
        
        result = load_totalseg_mask(self.totalseg_dir, case_name)
        
        if result is not None:
            self.totalseg_mask, self.totalseg_metadata = result
            self.state.totalseg_file = self.totalseg_metadata.get('source_file', 
                                                     ','.join(self.totalseg_metadata.get('source_files', [])))
            logger.info(f"Loaded TotalSegmentator mask: {self.totalseg_mask.shape}")
            logger.info(f"  Labels present: {np.unique(self.totalseg_mask)}")
        else:
            logger.info(f"No TotalSegmentator mask found for {case_name}")
    
    def _find_lv_aorta_junction(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Find the junction point between LV and Aorta from TotalSegmentator mask."""
        if self.totalseg_mask is None or self.totalseg_metadata is None:
            return None, None
        
        lv_label = TOTALSEG_LABELS['ventricle_left']
        aorta_label = TOTALSEG_LABELS['aorta']
        
        unique_labels = np.unique(self.totalseg_mask)
        if lv_label not in unique_labels or aorta_label not in unique_labels:
            return None, None
        
        lv_mask = (self.totalseg_mask == lv_label)
        aorta_mask = (self.totalseg_mask == aorta_label)
        
        struct = ndimage.generate_binary_structure(3, 1)
        lv_dilated = ndimage.binary_dilation(lv_mask, structure=struct, iterations=2)
        aorta_dilated = ndimage.binary_dilation(aorta_mask, structure=struct, iterations=2)
        
        junction = lv_dilated & aorta_dilated
        
        lv_surface = lv_mask & ~ndimage.binary_erosion(lv_mask, structure=struct)
        aorta_surface = aorta_mask & ~ndimage.binary_erosion(aorta_mask, structure=struct)
        
        junction = junction | (lv_surface & aorta_dilated) | (aorta_surface & lv_dilated)
        
        if not np.any(junction):
            return None, None
        
        junction_coords = np.array(np.where(junction))
        centroid_voxel = junction_coords.mean(axis=1)
        
        mask_origin = self.totalseg_metadata['origin']
        mask_spacing = self.totalseg_metadata['spacing']
        
        phys_x = mask_origin[0] + centroid_voxel[2] * mask_spacing[0]
        phys_y = mask_origin[1] + centroid_voxel[1] * mask_spacing[1]
        phys_z = mask_origin[2] + centroid_voxel[0] * mask_spacing[2]
        centroid_phys = np.array([phys_x, phys_y, phys_z])
        
        points_phys = np.zeros((junction_coords.shape[1], 3))
        points_phys[:, 0] = mask_origin[0] + junction_coords[2, :] * mask_spacing[0]
        points_phys[:, 1] = mask_origin[1] + junction_coords[1, :] * mask_spacing[1]
        points_phys[:, 2] = mask_origin[2] + junction_coords[0, :] * mask_spacing[2]
        
        lv_coords = np.array(np.where(lv_mask))
        aorta_coords = np.array(np.where(aorta_mask))
        
        self._lv_centroid = np.array([
            mask_origin[0] + lv_coords[2, :].mean() * mask_spacing[0],
            mask_origin[1] + lv_coords[1, :].mean() * mask_spacing[1],
            mask_origin[2] + lv_coords[0, :].mean() * mask_spacing[2]
        ])
        
        self._aorta_centroid = np.array([
            mask_origin[0] + aorta_coords[2, :].mean() * mask_spacing[0],
            mask_origin[1] + aorta_coords[1, :].mean() * mask_spacing[1],
            mask_origin[2] + aorta_coords[0, :].mean() * mask_spacing[2]
        ])
        
        return centroid_phys, points_phys
    
    def _compute_annulus_plane_from_junction(self, junction_points: np.ndarray) -> Optional[np.ndarray]:
        """Compute the annulus plane orientation from junction points using PCA.
        
        Orients the frame so that anterior (-Y in DICOM patient coordinates) is at the
        top of the axial view for standard radiological convention.
        """
        if junction_points is None or len(junction_points) < 10:
            return None
        
        centroid = junction_points.mean(axis=0)
        centered = junction_points - centroid
        
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        normal = eigenvectors[:, 0]
        in_plane_1 = eigenvectors[:, 2]
        in_plane_2 = eigenvectors[:, 1]
        
        # Ensure normal points from LV toward aorta (superior direction)
        if hasattr(self, '_lv_centroid') and hasattr(self, '_aorta_centroid'):
            lv_to_aorta = self._aorta_centroid - self._lv_centroid
            if np.dot(normal, lv_to_aorta) < 0:
                normal = -normal
        
        # DICOM patient coordinate system:
        # +X = patient left, +Y = patient posterior, +Z = patient superior
        # For axial view with anterior at top:
        # - The frame y-axis (up on screen) should point toward anterior (-Y in DICOM)
        # - The frame x-axis (right on screen) should point toward patient left (+X in DICOM)
        
        # Anterior direction in DICOM coords is -Y = [0, -1, 0]
        anterior_dir = np.array([0.0, -1.0, 0.0])
        
        # Project anterior direction onto the annulus plane
        anterior_in_plane = anterior_dir - np.dot(anterior_dir, normal) * normal
        anterior_in_plane_len = np.linalg.norm(anterior_in_plane)
        
        if anterior_in_plane_len > 0.1:
            # Use anterior as the y-axis (up direction in axial view)
            frame_y = anterior_in_plane / anterior_in_plane_len
            # x-axis is perpendicular to y and normal (right on screen)
            frame_x = np.cross(frame_y, normal)
            frame_x = frame_x / np.linalg.norm(frame_x)
        else:
            # Fallback if anterior is nearly parallel to normal
            frame_x = in_plane_1
            frame_y = in_plane_2
        
        # Ensure right-handed coordinate system
        cross = np.cross(frame_x, frame_y)
        if np.dot(cross, normal) < 0:
            frame_x = -frame_x
        
        frame = np.array([frame_x, frame_y, normal])
        
        for i in range(3):
            frame[i] = frame[i] / np.linalg.norm(frame[i])
        
        return frame
    
    def _auto_center_on_junction(self):
        """Auto-center the MPR view on the LV/Aorta junction and align to annulus plane."""
        centroid, junction_points = self._find_lv_aorta_junction()
        
        if centroid is not None:
            if (np.all(centroid >= self.mpr.phys_min) and 
                np.all(centroid <= self.mpr.phys_max)):
                self.mpr.intersection = centroid.copy()
                
                if junction_points is not None:
                    frame = self._compute_annulus_plane_from_junction(junction_points)
                    if frame is not None:
                        self.mpr.frame = frame
                
                return True
        
        return False
    
    def _get_case_name(self, case_path: Path) -> str:
        """Get the case name from a case Path."""
        return case_path.stem if self.npz_mode else case_path.name
    
    def _load_case(self, idx):
        if not 0 <= idx < len(self.cases):
            return
        
        self.current_idx = idx
        case = self.cases[idx]
        case_name = self._get_case_name(case)
        
        self._update_status(f"Loading {case_name}...")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
        try:
            if self.npz_mode:
                self.volume, spacing, origin = load_npz_volume(case)
            else:
                self.volume, spacing, origin = load_dicom_volume(case)
            self.mpr = MPREngine(self.volume, spacing, origin)
            
            self.state = CaseState(
                patient_id=case_name,
                spacing=spacing.tolist(),
                origin=origin.tolist()
            )
            
            state_file = self.output_dir / case_name / 'metadata.json'
            if state_file.exists():
                saved_state = CaseState.load(state_file)
                # Preserve spacing/origin from the volume file
                saved_state.spacing = spacing.tolist()
                saved_state.origin = origin.tolist()
                self.state = saved_state
            
            self._load_totalseg_mask(case_name)
            self._auto_center_on_junction()
            
            self._update_quality_buttons()
            self._update_valve_type_buttons()
            
            self.current_step = self.STEP_NAVIGATE
            self.preview_intersection = None
            self.crosshair_offset = np.array([0.0, 0.0])
            
            self.lvot_mask = None
            self.cusp_masks = [None, None, None]
            self.masks_visible = False
            
            self.annulus_nadir_idx = 0
            self.annulus_nadir_temp = []
            for artist in self.annulus_nadir_artists:
                artist.remove()
            self.annulus_nadir_artists = []
            
            self.ilt_ms_phase = 0
            self.ilt_ms_cc_temp = None
            self.ilt_ms_commissure_idx = 0
            self.ilt_ms_commissure_temp = []
            self.ilt_ms_nadir_temp = None
            self.ilt_ms_level = 0
            self.ilt_ms_level_points = []
            for artist in self.ilt_ms_artists:
                artist.remove()
            self.ilt_ms_artists = []
            
            self.lvot_septal_level = 0
            self.lvot_septal_level_points = []
            for artist in self.lvot_septal_artists:
                artist.remove()
            self.lvot_septal_artists = []
            
            self.mask_spacing = None
            self.mask_origin = None
            self.overall_mask = None
            
            for i in range(3):
                if self.mask_overlays[i]:
                    self.mask_overlays[i].remove()
                    self.mask_overlays[i] = None
                if self.ilt_ms_mask_overlays[i]:
                    self.ilt_ms_mask_overlays[i].remove()
                    self.ilt_ms_mask_overlays[i] = None
                if self.totalseg_overlays[i]:
                    self.totalseg_overlays[i].remove()
                    self.totalseg_overlays[i] = None
            
            self._clear_plane_lines()
            
            self._update_all_views()
            self._update_button_colors()
            
            totalseg_status = "TotalSeg: Available" if self.totalseg_mask is not None else "TotalSeg: Not found"
            
            # Count completed cases for progress display
            n_complete = 0
            for c in self.cases:
                cn = self._get_case_name(c)
                sf = self.output_dir / cn / 'metadata.json'
                if sf.exists():
                    try:
                        st = CaseState.load(sf)
                        if st.ct_quality in ('poor', 'prior_valve'):
                            n_complete += 1
                        elif st.ct_quality == 'acceptable':
                            expected_ilt = st.expected_ilt_ms_point_count()
                            if (st.annulus_set and st.stj_set and st.ilt_ms_set and st.lvot_set and
                                len(st.ilt_ms_points) == expected_ilt and len(st.lvot_septal_boundary_points) == 12):
                                n_complete += 1
                    except Exception:
                        pass
            
            self._update_status(f"[{n_complete}/{len(self.cases)} done] {case_name} | {totalseg_status}")
            
        except Exception as e:
            self._update_status(f"Error: {e}")
            traceback.print_exc()
    
    def _update_all_views(self, dragging_panel=None, rotation_dragging=False):
        if self.mpr is None:
            return
        
        sw = self.mpr.slice_width
        sh = self.mpr.slice_height
        
        for i in range(3):
            if dragging_panel is not None and i == dragging_panel:
                center = self.mpr.intersection
                frame = self.drag_start_frame if rotation_dragging else None
            elif dragging_panel is not None:
                if rotation_dragging and self.preview_frame is not None:
                    center = self.mpr.intersection
                    frame = self.preview_frame
                elif self.preview_intersection is not None:
                    center = self.preview_intersection
                    frame = None
                else:
                    center = self.mpr.intersection
                    frame = None
            else:
                center = self.mpr.intersection
                frame = None
            
            slice_data = self.mpr.extract_slice(i, center_override=center, frame_override=frame)
            
            if self.slice_imgs[i] is None:
                self.slice_imgs[i] = self.ax_panels[i].imshow(
                    slice_data, cmap='gray', aspect='equal', origin='lower',
                    extent=[0, sw, 0, sh], vmin=0, vmax=1
                )
                self.ax_panels[i].set_xlim(0, sw)
                self.ax_panels[i].set_ylim(0, sh)
            else:
                self.slice_imgs[i].set_data(slice_data)
                self.slice_imgs[i].set_extent([0, sw, 0, sh])
                self.ax_panels[i].set_xlim(0, sw)
                self.ax_panels[i].set_ylim(0, sh)
            
            self._draw_crosshairs(i, is_dragging=(dragging_panel == i))
            
            self._update_totalseg_overlay(i, center_override=center if dragging_panel is not None else None,
                                          frame_override=frame)
            
            self._update_mask_overlay(i, center_override=center if dragging_panel is not None else None,
                                      frame_override=frame)
        
        self._draw_plane_lines()
        self._draw_annulus_nadir_points()
        self._draw_ilt_ms_points()
        self._draw_lvot_septal_points()
        self._draw_boundary_splines()
        self.fig.canvas.draw_idle()
    
    def _update_totalseg_overlay(self, panel_idx, center_override=None, frame_override=None):
        """Update TotalSegmentator mask overlay for a panel."""
        ax = self.ax_panels[panel_idx]
        
        if not self.totalseg_visible or self.totalseg_mask is None:
            if self.totalseg_overlays[panel_idx]:
                self.totalseg_overlays[panel_idx].remove()
                self.totalseg_overlays[panel_idx] = None
            return
        
        mask_slice = self._extract_totalseg_slice(panel_idx, center_override=center_override, frame_override=frame_override)
        
        sw = self.mpr.slice_width
        sh = self.mpr.slice_height
        
        overlay = np.zeros((sh, sw, 4))
        
        for label, color in TOTALSEG_COLORS.items():
            overlay[mask_slice == label] = color
        
        if self.totalseg_overlays[panel_idx] is None:
            self.totalseg_overlays[panel_idx] = ax.imshow(
                overlay, aspect='equal', origin='lower', extent=[0, sw, 0, sh], zorder=1
            )
        else:
            self.totalseg_overlays[panel_idx].set_data(overlay)
            self.totalseg_overlays[panel_idx].set_extent([0, sw, 0, sh])
    
    def _extract_totalseg_slice(self, panel_idx, center_override=None, frame_override=None):
        """Extract a slice from the TotalSegmentator mask."""
        if self.totalseg_mask is None or self.totalseg_metadata is None:
            return np.zeros((self.mpr.slice_height, self.mpr.slice_width))
        
        center = center_override if center_override is not None else self.mpr.intersection
        x_axis, y_axis, _ = self.mpr.get_plane_axes(panel_idx, frame_override)
        
        half_w = self.mpr.slice_width // 2
        half_h = self.mpr.slice_height // 2
        u = np.linspace(-half_w, half_w, self.mpr.slice_width) * self.mpr.output_spacing
        v = np.linspace(-half_h, half_h, self.mpr.slice_height) * self.mpr.output_spacing
        uu, vv = np.meshgrid(u, v, indexing='xy')
        
        coords_phys = (center[np.newaxis, np.newaxis, :] +
                       uu[:, :, np.newaxis] * x_axis +
                       vv[:, :, np.newaxis] * y_axis)
        
        mask_origin = self.totalseg_metadata['origin']
        mask_spacing = self.totalseg_metadata['spacing']
        
        coords_voxel = np.zeros_like(coords_phys)
        coords_voxel[:, :, 0] = (coords_phys[:, :, 2] - mask_origin[2]) / mask_spacing[2]
        coords_voxel[:, :, 1] = (coords_phys[:, :, 1] - mask_origin[1]) / mask_spacing[1]
        coords_voxel[:, :, 2] = (coords_phys[:, :, 0] - mask_origin[0]) / mask_spacing[0]
        
        return ndimage.map_coordinates(
            self.totalseg_mask.astype(float),
            [coords_voxel[:, :, 0].ravel(),
             coords_voxel[:, :, 1].ravel(),
             coords_voxel[:, :, 2].ravel()],
            order=0, mode='constant', cval=0
        ).reshape(self.mpr.slice_height, self.mpr.slice_width)
    
    def _draw_crosshairs(self, panel_idx, is_dragging=False):
        ax = self.ax_panels[panel_idx]
        
        for artist in self.crosshair_artists[panel_idx]:
            artist.remove()
        self.crosshair_artists[panel_idx] = []
        
        if self.current_step != self.STEP_NAVIGATE:
            return
        
        sw = self.mpr.slice_width
        sh = self.mpr.slice_height
        cx, cy = sw / 2, sh / 2
        
        h_color, v_color = self.mpr.get_crosshair_colors(panel_idx)
        
        if is_dragging and self.drag_type == 'translate':
            cx += self.crosshair_offset[0]
            cy += self.crosshair_offset[1]
        
        in_plane_rotation = 0.0
        if is_dragging and self.drag_type == 'rotate':
            in_plane_rotation = self.total_rotation_angle
        
        h_angle_rad = np.radians(in_plane_rotation)
        h_len = sw
        hx1 = cx - h_len * np.cos(h_angle_rad)
        hy1 = cy - h_len * np.sin(h_angle_rad)
        hx2 = cx + h_len * np.cos(h_angle_rad)
        hy2 = cy + h_len * np.sin(h_angle_rad)
        h_line, = ax.plot([hx1, hx2], [hy1, hy2], color=h_color, linewidth=1.5, alpha=0.8)
        
        v_angle_rad = np.radians(90 + in_plane_rotation)
        v_len = sh
        vx1 = cx - v_len * np.cos(v_angle_rad)
        vy1 = cy - v_len * np.sin(v_angle_rad)
        vx2 = cx + v_len * np.cos(v_angle_rad)
        vy2 = cy + v_len * np.sin(v_angle_rad)
        v_line, = ax.plot([vx1, vx2], [vy1, vy2], color=v_color, linewidth=1.5, alpha=0.8)
        
        center, = ax.plot(cx, cy, 'o', color='yellow', markersize=6, 
                          markerfacecolor='none', markeredgewidth=1.5)
        
        self.crosshair_artists[panel_idx] = [h_line, v_line, center]
    
    def _update_mask_overlay(self, panel_idx, center_override=None, frame_override=None):
        """Update generated masks overlay for a panel."""
        ax = self.ax_panels[panel_idx]
        sw = self.mpr.slice_width
        sh = self.mpr.slice_height
        
        def extract_mask_slice_totalseg(mask, panel_idx):
            if mask is None:
                return np.zeros((sh, sw))
            
            center = center_override if center_override is not None else self.mpr.intersection
            x_axis, y_axis, _ = self.mpr.get_plane_axes(panel_idx, frame_override)
            
            half_w = sw // 2
            half_h = sh // 2
            u = np.linspace(-half_w, half_w, sw) * self.mpr.output_spacing
            v = np.linspace(-half_h, half_h, sh) * self.mpr.output_spacing
            uu, vv = np.meshgrid(u, v, indexing='xy')
            
            coords_phys = (center[np.newaxis, np.newaxis, :] +
                          uu[:, :, np.newaxis] * x_axis +
                          vv[:, :, np.newaxis] * y_axis)
            
            voxel_z = (coords_phys[:, :, 2] - self.mask_origin[2]) / self.mask_spacing[2]
            voxel_y = (coords_phys[:, :, 1] - self.mask_origin[1]) / self.mask_spacing[1]
            voxel_x = (coords_phys[:, :, 0] - self.mask_origin[0]) / self.mask_spacing[0]
            
            vk = np.clip(np.round(voxel_z).astype(int), 0, mask.shape[0] - 1)
            vj = np.clip(np.round(voxel_y).astype(int), 0, mask.shape[1] - 1)
            vi = np.clip(np.round(voxel_x).astype(int), 0, mask.shape[2] - 1)
            
            return mask[vk, vj, vi]
        
        # Main masks overlay (LVOT + cusps)
        if not self.masks_visible or not hasattr(self, 'mask_spacing') or self.mask_spacing is None:
            if self.mask_overlays[panel_idx]:
                self.mask_overlays[panel_idx].remove()
                self.mask_overlays[panel_idx] = None
        else:
            overlay = np.zeros((sh, sw, 4))
            
            if self.lvot_mask is not None:
                lvot_slice = extract_mask_slice_totalseg(self.lvot_mask, panel_idx)
                overlay[lvot_slice > 0] = [1, 0, 1, 0.3]  # Magenta
            
            cusp_colors = [
                [1, 0, 0, 0.3],  # Red
                [0, 1, 0, 0.3],  # Green
                [0, 0, 1, 0.3],  # Blue
            ]
            for i, cusp_mask in enumerate(self.cusp_masks):
                if cusp_mask is not None:
                    cusp_slice = extract_mask_slice_totalseg(cusp_mask, panel_idx)
                    overlay[cusp_slice > 0] = cusp_colors[i]
            
            if self.mask_overlays[panel_idx] is None:
                self.mask_overlays[panel_idx] = ax.imshow(
                    overlay, aspect='equal', origin='lower', extent=[0, sw, 0, sh], zorder=2
                )
            else:
                self.mask_overlays[panel_idx].set_data(overlay)
                self.mask_overlays[panel_idx].set_extent([0, sw, 0, sh])
        
        # ILT/MS surface mask overlay (yellow)
        if not self.ilt_ms_mask_visible or self.ms_surface_mask is None or not hasattr(self, 'mask_spacing') or self.mask_spacing is None:
            if self.ilt_ms_mask_overlays[panel_idx]:
                self.ilt_ms_mask_overlays[panel_idx].remove()
                self.ilt_ms_mask_overlays[panel_idx] = None
        else:
            overlay = np.zeros((sh, sw, 4))
            ms_slice = extract_mask_slice_totalseg(self.ms_surface_mask, panel_idx)
            overlay[ms_slice > 0] = [1, 1, 0, 0.5]  # Yellow
            
            if self.ilt_ms_mask_overlays[panel_idx] is None:
                self.ilt_ms_mask_overlays[panel_idx] = ax.imshow(
                    overlay, aspect='equal', origin='lower', extent=[0, sw, 0, sh], zorder=3
                )
            else:
                self.ilt_ms_mask_overlays[panel_idx].set_data(overlay)
                self.ilt_ms_mask_overlays[panel_idx].set_extent([0, sw, 0, sh])
    
    def _clear_plane_lines(self):
        for artist in self.plane_line_artists:
            artist.remove()
        self.plane_line_artists = []
        for artist in self.current_z_artists:
            artist.remove()
        self.current_z_artists = []
    
    def _compute_plane_line_endpoints(self, plane, panel_idx):
        """Compute the actual intersection line of a 3D plane with the 2D panel slice.
        
        Returns (x1, y1, x2, y2) in pixel coordinates, or None if no intersection.
        """
        if plane is None:
            return None
        
        plane_center = plane.get_center_array()
        plane_normal = plane.get_normal_array()
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        
        x_axis, y_axis, z_axis = self.mpr.get_plane_axes(panel_idx)
        
        sw = self.mpr.slice_width
        sh = self.mpr.slice_height
        
        # The panel slice is defined by points: intersection + u*x_axis + v*y_axis
        # The 3D plane is defined by: dot(P - plane_center, plane_normal) = 0
        # 
        # For a point on the panel slice:
        # dot(intersection + u*x_axis + v*y_axis - plane_center, plane_normal) = 0
        # dot(intersection - plane_center, plane_normal) + u*dot(x_axis, plane_normal) + v*dot(y_axis, plane_normal) = 0
        #
        # Let: d = dot(intersection - plane_center, plane_normal)
        #      a = dot(x_axis, plane_normal)
        #      b = dot(y_axis, plane_normal)
        # Then: d + a*u + b*v = 0  =>  v = -(d + a*u) / b  (if b != 0)
        
        d = np.dot(self.mpr.intersection - plane_center, plane_normal)
        a = np.dot(x_axis, plane_normal)
        b = np.dot(y_axis, plane_normal)
        
        # Convert pixel to physical offset
        # u_mm = (x_px - sw/2) * output_spacing
        # v_mm = (y_px - sh/2) * output_spacing
        
        # If b is near zero, the plane is parallel to the y-axis (vertical line)
        # If a is near zero, the plane is parallel to the x-axis (horizontal line)
        
        if abs(b) < 1e-10 and abs(a) < 1e-10:
            # Plane is parallel to the slice - no intersection line
            return None
        
        if abs(b) < 1e-10:
            # Plane is vertical (parallel to y-axis)
            # a*u + d = 0  =>  u = -d/a
            u_mm = -d / a
            x_px = sw / 2 + u_mm / self.mpr.output_spacing
            if 0 <= x_px <= sw:
                return (x_px, 0, x_px, sh)
            return None
        
        # General case: compute y at left edge (x=0) and right edge (x=sw)
        # x_px = 0  =>  u_mm = (0 - sw/2) * output_spacing = -sw/2 * output_spacing
        # x_px = sw =>  u_mm = (sw - sw/2) * output_spacing = sw/2 * output_spacing
        
        u_left = -sw / 2 * self.mpr.output_spacing
        u_right = sw / 2 * self.mpr.output_spacing
        
        v_left = -(d + a * u_left) / b
        v_right = -(d + a * u_right) / b
        
        y_left = sh / 2 + v_left / self.mpr.output_spacing
        y_right = sh / 2 + v_right / self.mpr.output_spacing
        
        return (0, y_left, sw, y_right)
    
    def _draw_plane_lines(self):
        self._clear_plane_lines()
        
        if self.mpr is None or self.state is None:
            return
        
        def draw_plane_line(plane, color, name, panel_idx):
            endpoints = self._compute_plane_line_endpoints(plane, panel_idx)
            
            if endpoints is None:
                return
            
            x1, y1, x2, y2 = endpoints
            
            # Check if line is reasonably on screen
            sh = self.mpr.slice_height
            if (y1 < -sh and y2 < -sh) or (y1 > 2*sh and y2 > 2*sh):
                return
            
            line, = self.ax_panels[panel_idx].plot([x1, x2], [y1, y2],
                                                    color=color, linewidth=2, alpha=0.8)
            line.set_gid(name)
            self.plane_line_artists.append(line)
        
        if self.state.annulus_plane is not None:
            if self.state.annulus_set:
                draw_plane_line(self.state.annulus_plane, 'orange', 'annulus', 0)
                draw_plane_line(self.state.annulus_plane, 'orange', 'annulus', 1)
        
        if self.state.stj_plane is not None:
            if self.state.stj_set or self.current_step == self.STEP_STJ:
                draw_plane_line(self.state.stj_plane, 'cyan', 'stj', 0)
                draw_plane_line(self.state.stj_plane, 'cyan', 'stj', 1)
        
        if self.state.lvot_plane is not None:
            if self.state.lvot_set or (self.current_step == self.STEP_ILT_MS and self.ilt_ms_phase == 4):
                draw_plane_line(self.state.lvot_plane, 'magenta', 'lvot', 0)
                draw_plane_line(self.state.lvot_plane, 'magenta', 'lvot', 1)
    
    def _update_status(self, msg):
        self.status_text.set_text(msg)
        self.fig.canvas.draw_idle()
    
    def _get_panel_idx(self, event):
        for i, ax in enumerate(self.ax_panels):
            if event.inaxes == ax:
                return i
        return None
    
    def _get_click_zone(self, x, y):
        sw = self.mpr.slice_width
        sh = self.mpr.slice_height
        cx, cy = sw / 2, sh / 2
        
        dist_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        if dist_center < min(sw, sh) * 0.15:
            return 'center'
        
        if abs(y - cy) < 20 and abs(x - cx) > sw * 0.15:
            return 'rotate'
        
        if abs(x - cx) < 20 and abs(y - cy) > sh * 0.15:
            return 'rotate'
        
        return 'center'
    
    def _check_plane_line_click(self, panel_idx, x, y):
        """Check if click is on a plane line and return (plane_name, mode)."""
        if panel_idx not in [0, 1]:
            return None, None
        
        sw = self.mpr.slice_width
        
        def check_line(plane, name, allow_rotate=True):
            if plane is None:
                return None, None
            
            endpoints = self._compute_plane_line_endpoints(plane, panel_idx)
            
            if endpoints is None:
                return None, None
            
            x1, y1, x2, y2 = endpoints
            
            # Compute y position on line at click x
            if abs(x2 - x1) < 1e-10:
                # Vertical line
                if abs(x - x1) < 15:
                    return name, 'translate'
                return None, None
            
            t = (x - x1) / (x2 - x1)
            y_at_x = y1 + t * (y2 - y1)
            
            if abs(y - y_at_x) < 15:
                if abs(x - sw/2) < sw * 0.25:
                    return name, 'translate'
                elif allow_rotate:
                    return name, 'rotate'
                else:
                    return name, 'translate'
            return None, None
        
        if self.current_step == self.STEP_STJ:
            return check_line(self.state.stj_plane, 'stj', allow_rotate=True)
        
        return None, None
    
    def _on_press(self, event):
        if event.inaxes is None:
            return
        
        panel = self._get_panel_idx(event)
        
        # Annulus nadir point placement on axial panel
        if self.current_step == self.STEP_ANNULUS and panel == 2:
            self._add_annulus_nadir_point(event.xdata, event.ydata)
            return
        
        # ILT/MS/LVOT point placement on axial panel
        if self.current_step == self.STEP_ILT_MS and panel == 2:
            self._add_ilt_ms_point(event.xdata, event.ydata)
            return
        
        # Check for plane line dragging (STJ only - annulus is set by 3 points, LVOT is auto-calculated)
        if self.current_step == self.STEP_STJ and panel in [0, 1]:
            line_name, mode = self._check_plane_line_click(panel, event.xdata, event.ydata)
            if line_name:
                self.dragging = True
                self.drag_type = f'{line_name}_{mode[0]}'
                self.drag_panel = panel
                self.drag_start_mouse = (event.xdata, event.ydata)
                
                plane = self.state.stj_plane
                
                self.drag_start_center = plane.get_center_array().copy()
                self.drag_start_normal = plane.get_normal_array().copy()
                
                if mode == 'rotate':
                    sw = self.mpr.slice_width
                    endpoints = self._compute_plane_line_endpoints(plane, panel)
                    if endpoints is not None:
                        x1, y1, x2, y2 = endpoints
                        # Compute y at center of screen
                        if abs(x2 - x1) > 1e-10:
                            t = (sw/2 - x1) / (x2 - x1)
                            y_center = y1 + t * (y2 - y1)
                        else:
                            y_center = (y1 + y2) / 2
                        self.plane_last_drag_angle = np.arctan2(event.ydata - y_center, event.xdata - sw/2)
                    else:
                        self.plane_last_drag_angle = None
                    self.plane_total_rotation = 0.0
                
                return
        
        if self.current_step == self.STEP_NAVIGATE and panel is not None:
            zone = self._get_click_zone(event.xdata, event.ydata)
            
            self.dragging = True
            self.drag_panel = panel
            self.drag_start_mouse = (event.xdata, event.ydata)
            self.drag_start_intersection = self.mpr.intersection.copy()
            self.preview_intersection = self.mpr.intersection.copy()
            self.crosshair_offset = np.array([0.0, 0.0])
            
            self.drag_start_frame = self.mpr.copy_frame()
            self.preview_frame = None
            self.total_rotation_angle = 0.0
            
            if zone == 'center':
                self.drag_type = 'translate'
            else:
                self.drag_type = 'rotate'
                sw = self.mpr.slice_width
                sh = self.mpr.slice_height
                cx, cy = sw / 2, sh / 2
                self.last_drag_angle = np.arctan2(event.ydata - cy, event.xdata - cx)
    
    def _on_release(self, event):
        if self.dragging and self.drag_type == 'translate':
            if self.preview_intersection is not None:
                self.mpr.intersection = self.preview_intersection.copy()
            self.crosshair_offset = np.array([0.0, 0.0])
            self.preview_intersection = None
            self._update_all_views()
        
        elif self.dragging and self.drag_type == 'rotate':
            if self.preview_frame is not None:
                self.mpr.frame = self.preview_frame.copy()
            self.total_rotation_angle = 0.0
            self.preview_frame = None
            self.drag_start_frame = None
            self.last_drag_angle = None
            self._update_all_views()
        
        elif self.dragging and self.drag_type in ['stj_t', 'stj_r']:
            self._update_button_colors()
        
        self.dragging = False
        self.drag_type = None
        self.drag_panel = None
        self.drag_start_mouse = None
        self.drag_start_intersection = None
        self.preview_intersection = None
        self.drag_start_center = None
        self.drag_start_normal = None
    
    def _on_motion(self, event):
        if event.inaxes is None:
            return
        
        if not self.dragging:
            return
        
        if self.drag_type in ['stj_t', 'stj_r']:
            sw = self.mpr.slice_width
            
            plane = self.state.stj_plane
            
            x_axis, y_axis, z_axis = self.mpr.get_plane_axes(self.drag_panel)
            
            if self.drag_type.endswith('_t'):
                dy = event.ydata - self.drag_start_mouse[1]
                delta_mm = dy * self.mpr.output_spacing
                
                new_center = self.drag_start_center + delta_mm * y_axis
                plane.center = new_center.tolist()
            else:
                endpoints = self._compute_plane_line_endpoints(plane, self.drag_panel)
                
                if endpoints is not None:
                    x1, y1, x2, y2 = endpoints
                    # Compute y at center of screen
                    if abs(x2 - x1) > 1e-10:
                        t = (sw/2 - x1) / (x2 - x1)
                        y_center = y1 + t * (y2 - y1)
                    else:
                        y_center = (y1 + y2) / 2
                    
                    current_angle = np.arctan2(event.ydata - y_center, event.xdata - sw/2)
                    
                    if self.plane_last_drag_angle is not None:
                        delta = current_angle - self.plane_last_drag_angle
                        delta = wrap_angle(np.degrees(delta))
                        self.plane_total_rotation += delta
                    
                    self.plane_last_drag_angle = current_angle
                    
                    rotation_angle = ROTATION_SIGN[self.drag_panel] * self.plane_total_rotation
                    rotation_rad = np.radians(rotation_angle)
                    
                    k = z_axis / np.linalg.norm(z_axis)
                    v = self.drag_start_normal
                    new_normal = (v * np.cos(rotation_rad) + 
                                  np.cross(k, v) * np.sin(rotation_rad) + 
                                  k * np.dot(k, v) * (1 - np.cos(rotation_rad)))
                    new_normal = new_normal / np.linalg.norm(new_normal)
                    plane.normal = new_normal.tolist()
            
            self._draw_plane_lines()
            self.fig.canvas.draw_idle()
            return
        
        if self.drag_type == 'translate':
            dx = event.xdata - self.drag_start_mouse[0]
            dy = event.ydata - self.drag_start_mouse[1]
            
            self.crosshair_offset = np.array([dx, dy])
            
            offset_phys = self.mpr.pixel_to_physical_offset(self.drag_panel, dx, dy)
            self.preview_intersection = self.drag_start_intersection + offset_phys
            self.preview_intersection = np.clip(self.preview_intersection,
                                                 self.mpr.phys_min, self.mpr.phys_max)
            
            self._update_all_views(dragging_panel=self.drag_panel)
            return
        
        if self.drag_type == 'rotate':
            sw = self.mpr.slice_width
            sh = self.mpr.slice_height
            cx, cy = sw / 2, sh / 2
            
            current_angle = np.arctan2(event.ydata - cy, event.xdata - cx)
            
            if self.last_drag_angle is not None:
                delta = current_angle - self.last_drag_angle
                delta = wrap_angle(np.degrees(delta))
                self.total_rotation_angle += delta
            
            self.last_drag_angle = current_angle
            
            _, _, normal = self.mpr.get_plane_axes(self.drag_panel, self.drag_start_frame)
            
            rotation_angle = ROTATION_SIGN[self.drag_panel] * self.total_rotation_angle
            self.preview_frame = MPREngine.rotate_frame_by_axis(
                self.drag_start_frame, normal, rotation_angle
            )
            
            self._update_all_views(dragging_panel=self.drag_panel, rotation_dragging=True)
    
    def _on_scroll(self, event):
        if event.inaxes is None or self.mpr is None:
            return
        
        panel = self._get_panel_idx(event)
        if panel is not None:
            _, _, normal = self.mpr.get_plane_axes(panel)
            delta = -event.step * 2 * self.mpr.output_spacing
            self.mpr.intersection += delta * normal
            self.mpr.intersection = np.clip(self.mpr.intersection,
                                            self.mpr.phys_min, self.mpr.phys_max)
            self._update_all_views()
    
    def _on_key(self, event):
        if event.key == 'enter':
            self._save_and_next()
        elif event.key == 'escape':
            self.current_step = self.STEP_NAVIGATE
            self._update_all_views()
            self._update_button_colors()
    
    def _export_bspline_surface_geomdl(self, surface_params, filepath):
        """Export BSplineBoundarySurface to geomdl JSON format.
        
        This creates a standard B-spline surface that anyone can load with geomdl.
        """
        try:
            control_points = surface_params['control_points']
            t_values = surface_params['t_values']
            
            # Create geomdl BSpline.Surface
            surf = GeomdlBSpline.Surface()
            
            # Flatten control points for geomdl (expects 1D list)
            n_u = len(control_points)  # number of t-levels
            n_v = len(control_points[0])  # points per level (4)
            
            flat_ctrlpts = []
            for level in control_points:
                for pt in level:
                    flat_ctrlpts.append(pt)
            
            surf.degree_u = min(3, n_u - 1)
            surf.degree_v = min(3, n_v - 1)
            surf.set_ctrlpts(flat_ctrlpts, n_u, n_v)
            surf.knotvector_u = geomdl_utilities.generate_knot_vector(surf.degree_u, n_u)
            surf.knotvector_v = geomdl_utilities.generate_knot_vector(surf.degree_v, n_v)
            
            # Export to JSON
            geomdl_exchange.export_json(surf, str(filepath))
            return True
        except Exception as e:
            logger.error(f"Error exporting B-spline surface: {e}")
            return False
    
    def _export_surface_vtk_mesh(self, surface_params, filepath, n_t=40, n_angle=60):
        """Export BSplineBoundarySurface as VTK mesh for visualization."""
        try:
            # Reconstruct surface
            surface = BSplineBoundarySurface.from_dict(surface_params)
            
            t_min, t_max = surface.t_values[0], surface.t_values[-1]
            t_samples = np.linspace(t_min, t_max, n_t)
            
            # Collect all points
            points = []
            for t in t_samples:
                contour = surface.get_contour_at_t(t, n_points=n_angle)
                for pt in contour:
                    points.append(pt)
            
            points = np.array(points)
            
            # Build triangles (connect adjacent t-levels)
            triangles = []
            for i in range(n_t - 1):
                for j in range(n_angle - 1):
                    # Current level indices
                    p0 = i * n_angle + j
                    p1 = i * n_angle + j + 1
                    # Next level indices
                    p2 = (i + 1) * n_angle + j
                    p3 = (i + 1) * n_angle + j + 1
                    
                    # Two triangles per quad
                    triangles.append([p0, p1, p2])
                    triangles.append([p1, p3, p2])
            
            # Write VTK file (legacy ASCII format)
            with open(filepath, 'w') as f:
                f.write("# vtk DataFile Version 3.0\n")
                f.write("BSpline Surface Mesh\n")
                f.write("ASCII\n")
                f.write("DATASET POLYDATA\n")
                f.write(f"POINTS {len(points)} float\n")
                for pt in points:
                    f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")
                f.write(f"POLYGONS {len(triangles)} {len(triangles) * 4}\n")
                for tri in triangles:
                    f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
            
            return True
        except Exception as e:
            logger.error(f"Error exporting VTK mesh: {e}")
            return False
    
    def _print_demo_summary(self, case_name: str):
        """Print annotation measurements to console (demo mode)."""
        s = self.state
        print("\n" + "=" * 60)
        print(f"  ANNOTATION SUMMARY: {case_name}")
        print("=" * 60)
        print(f"  Valve type:   {s.valve_type or 'not set'}")
        print(f"  CT quality:   {s.ct_quality or 'not set'}")
        
        if s.annulus_set and len(s.annulus_nadirs) == 3:
            nadirs = [np.array(n) for n in s.annulus_nadirs]
            center = np.array(s.annulus_plane.center)
            
            # Intercommissural distances (nadir to nadir)
            labels = ['R', 'L', 'N']
            pairs = [(0, 1, 'R-L'), (1, 2, 'L-N'), (0, 2, 'R-N')]
            print(f"\n  Annulus plane center: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}]")
            print(f"  Nadir distances:")
            for i, j, label in pairs:
                d = np.linalg.norm(nadirs[i] - nadirs[j])
                print(f"    {label}: {d:.1f} mm")
            
            # Nadir depths (perpendicular distance from annular plane)
            normal = np.array(s.annulus_plane.normal)
            print(f"  Nadir depths (below annular plane):")
            for i, label in enumerate(labels):
                depth = np.dot(nadirs[i] - center, normal)
                print(f"    {label}: {abs(depth):.1f} mm")
            
            # Annular plane tilt (angle from horizontal)
            z_axis = np.array([0, 0, 1])
            tilt_deg = np.degrees(np.arccos(np.clip(abs(np.dot(normal, z_axis)), 0, 1)))
            print(f"  Annular plane tilt: {tilt_deg:.1f} deg from axial")
        
        if s.stj_set:
            stj_center = np.array(s.stj_plane.center)
            ann_center = np.array(s.annulus_plane.center)
            height = np.linalg.norm(stj_center - ann_center)
            print(f"\n  STJ center: [{stj_center[0]:.1f}, {stj_center[1]:.1f}, {stj_center[2]:.1f}]")
            print(f"  Annulus-to-STJ distance: {height:.1f} mm")
        
        if s.ilt_ms_set and len(s.commissure_points) == 3:
            comm = [np.array(c) for c in s.commissure_points]
            comm_labels = ['RN', 'LR', 'NL']
            print(f"\n  Intercommissural distances:")
            for i in range(3):
                j = (i + 1) % 3
                d = np.linalg.norm(comm[i] - comm[j])
                print(f"    {comm_labels[i]}-{comm_labels[j]}: {d:.1f} mm")
        
        if s.lvot_set:
            print(f"  LVOT plane: set")
        
        n_ilt = len(s.ilt_ms_points) if s.ilt_ms_points else 0
        n_lvot_sep = len(s.lvot_septal_boundary_points) if s.lvot_septal_boundary_points else 0
        print(f"\n  ILT/MS boundary points: {n_ilt}")
        print(f"  LVOT septal boundary points: {n_lvot_sep}")
        print("=" * 60 + "\n")
    
    def _save_and_next(self):
        case_name = self._get_case_name(self.cases[self.current_idx])
        out = self.output_dir / case_name
        out.mkdir(parents=True, exist_ok=True)
        
        try:
            if self.mask_spacing is not None:
                mask_spacing = [float(self.mask_spacing[0]), 
                               float(self.mask_spacing[1]), 
                               float(self.mask_spacing[2])]
                mask_origin = [float(self.mask_origin[0]),
                              float(self.mask_origin[1]),
                              float(self.mask_origin[2])]
            else:
                mask_spacing = [float(self.state.spacing[2]),
                               float(self.state.spacing[1]),
                               float(self.state.spacing[0])]
                mask_origin = [float(o) for o in self.state.origin]
            
            def save_mask(mask, filename):
                if mask is not None and np.sum(mask) > 0:
                    img = sitk.GetImageFromArray(mask.astype(np.uint8))
                    img.SetSpacing(mask_spacing)
                    img.SetOrigin(mask_origin)
                    sitk.WriteImage(img, str(out / filename))
            
            # Overall and LVOT masks
            save_mask(self.overall_mask, 'overall_mask.nii.gz')
            save_mask(self.lvot_mask, 'lvot_mask.nii.gz')
            
            # MS surface mask
            save_mask(self.ms_surface_mask, 'ms_surface_mask.nii.gz')
            
            # Save cusp masks with appropriate names based on valve type
            if self.state.is_type0_bicuspid():
                cusp_names = ['cusp_A', 'cusp_P']
                for i in range(2):
                    save_mask(self.cusp_masks[i], f'{cusp_names[i]}_mask.nii.gz')
            else:
                cusp_names = ['cusp_R', 'cusp_L', 'cusp_N']
                for i in range(3):
                    save_mask(self.cusp_masks[i], f'{cusp_names[i]}_mask.nii.gz')
            
            self.state.mpr_frame = self.mpr.frame.tolist() if self.mpr else None
            self.state.save(out / 'metadata.json')
            
            # Export B-spline surfaces (geomdl JSON + VTK mesh)
            if self.state.ilt_ms_surface_params:
                self._export_bspline_surface_geomdl(
                    self.state.ilt_ms_surface_params, out / 'ilt_ms_surface.json')
                self._export_surface_vtk_mesh(
                    self.state.ilt_ms_surface_params, out / 'ilt_ms_surface.vtk')
            
            logger.info(f"Saved {case_name}")
            self._update_status(f"Saved {case_name}")
            
            if self.demo:
                self._print_demo_summary(case_name)
            
            self._next_case()
            
        except Exception as e:
            self._update_status(f"Error: {e}")
            traceback.print_exc()
    
    def _next_case(self):
        for i in range(self.current_idx + 1, len(self.cases)):
            if self.valve_type_filter or self.poor_only:
                state_file = self.output_dir / self._get_case_name(self.cases[i]) / 'metadata.json'
                if not state_file.exists():
                    continue
                try:
                    existing_state = CaseState.load(state_file)
                    if self.valve_type_filter and existing_state.valve_type != self.valve_type_filter:
                        continue
                    if self.poor_only and existing_state.ct_quality != 'poor':
                        continue
                except Exception:
                    continue
            self._load_case(i)
            return
    
    def _prev_case(self):
        for i in range(self.current_idx - 1, -1, -1):
            if self.valve_type_filter or self.poor_only:
                state_file = self.output_dir / self._get_case_name(self.cases[i]) / 'metadata.json'
                if not state_file.exists():
                    continue
                try:
                    existing_state = CaseState.load(state_file)
                    if self.valve_type_filter and existing_state.valve_type != self.valve_type_filter:
                        continue
                    if self.poor_only and existing_state.ct_quality != 'poor':
                        continue
                except Exception:
                    continue
            self._load_case(i)
            return
    
    def _save_poor_quality_and_next(self):
        """Save minimal metadata for poor quality CT and move to next case."""
        case_name = self._get_case_name(self.cases[self.current_idx])
        out = self.output_dir / case_name
        out.mkdir(parents=True, exist_ok=True)
        
        minimal_state = CaseState(
            patient_id=self.state.patient_id,
            ct_quality='poor',
            spacing=self.state.spacing,
            origin=self.state.origin,
        )
        minimal_state.save(out / 'metadata.json')
        
        logger.info(f"Saved {case_name} as poor quality")
        self._update_status(f"Saved {case_name} as poor quality - moving to next")
        self._next_case()


def main():
    parser = argparse.ArgumentParser(description='Aortic Root MPR Segmentation Tool')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing .npz files or case folders with DICOM data')
    parser.add_argument('--output_dir', type=str, default='data/segmentations',
                        help='Directory for saving segmentation outputs (default: data/segmentations)')
    parser.add_argument('--totalseg_dir', type=str, default=None,
                        help='Directory containing TotalSegmentator outputs')
    parser.add_argument('--csv', type=str, default=None,
                        help='CSV file with patient data for filtering (optional)')
    parser.add_argument('--id_column', type=str, default='mrn',
                        help='Column name for patient ID in CSV (default: mrn)')
    parser.add_argument('--exclude_column', type=str, default=None,
                        help='Column name for exclusion criteria (e.g., PriorAorticValve)')
    parser.add_argument('--exclude_value', type=str, default=None,
                        help='Value that triggers exclusion (e.g., Yes)')
    parser.add_argument('--reprocess', action='store_true',
                        help='Reprocess all cases, including those with existing metadata.json')
    parser.add_argument('--valve_type', type=str, default=None,
                        help='Filter to cases with specific valve_type (e.g., type0, tricuspid)')
    parser.add_argument('--poor_only', action='store_true',
                        help='Filter to only show cases previously marked as poor quality')
    parser.add_argument('--demo', action='store_true',
                        help='Demo mode: saves to data/segmentations_demo/, prints measurements on save')
    parser.add_argument('--case', type=str, default=None,
                        help='Load a single case by name (e.g., case_001)')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if args.demo:
        output_dir = Path('data/segmentations_demo')
        logger.info(f"Demo mode: saving to {output_dir}")
    else:
        output_dir = Path(args.output_dir)
    
    totalseg_dir = Path(args.totalseg_dir) if args.totalseg_dir else None
    
    if not data_dir.exists():
        logger.error(f"Not found: {data_dir}")
        return
    
    AorticSegmentor(
        data_dir, 
        output_dir, 
        totalseg_dir,
        csv_file=Path(args.csv) if args.csv else None,
        id_column=args.id_column,
        exclude_column=args.exclude_column,
        exclude_value=args.exclude_value,
        reprocess=args.reprocess,
        valve_type_filter=args.valve_type,
        poor_only=args.poor_only,
        demo=args.demo,
        case_filter=args.case,
    ).run()


if __name__ == '__main__':
    main()