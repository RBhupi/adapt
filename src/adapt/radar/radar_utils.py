#!/usr/bin/env python3
"""Radar utilities for cell analysis and projection.

Centralized utilities for:
- Grid spacing and time interval computation
- Cell centroids and geometry
- Projection vectors
"""

import numpy as np
import xarray as xr
import logging
from typing import Dict, Tuple, Optional

try:
    from scipy.ndimage import center_of_mass, label as scipy_label
    from skimage.morphology import convex_hull_image, binary_closing
except ImportError:
    scipy_label = None    
    center_of_mass = None
    convex_hull_image = None
    binary_closing = None

logger = logging.getLogger(__name__)


# ============================================================================
# GRID AND TIME UTILITIES
# ============================================================================

def compute_time_interval(ds1: xr.Dataset, ds2: xr.Dataset, units: str = 'seconds') -> float:
    """Compute time interval between two datasets.
    
    Parameters
    ----------
    ds1, ds2 : xr.Dataset
        Datasets with time coordinate
    units : str
        'seconds' or 'minutes'
    
    Returns
    -------
    float
        Time interval in specified units
    """
    t1 = ds1.time.values[0]
    t2 = ds2.time.values[0]
    
    if units == 'seconds':
        return float((t2 - t1) / np.timedelta64(1, 's'))
    elif units == 'minutes':
        return float((t2 - t1) / np.timedelta64(1, 'm'))
    else:
        raise ValueError(f"Unsupported units: {units}")


def get_grid_spacing(ds: xr.Dataset) -> Tuple[float, float]:
    """Get grid spacing from xarray dataset coordinates.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset with x, y coordinates
    
    Returns
    -------
    spacing_x, spacing_y : tuple of float
        Grid spacing in meters
    """
    spacing_x = float((ds.x[-1] - ds.x[0]) / (len(ds.x) - 1))
    spacing_y = float((ds.y[-1] - ds.y[0]) / (len(ds.y) - 1))
    return spacing_x, spacing_y


# ============================================================================
# CENTROID COMPUTATION
# ============================================================================

def compute_cell_centroid(label_array: np.ndarray, label_id: int,
                         weighted_by: np.ndarray = None) -> Optional[Tuple[float, float]]:
    """Compute centroid of a labeled region.
    
    Parameters
    ----------
    label_array : np.ndarray
        2D array of integer labels.
    label_id : int
        Label ID to find centroid for (must be > 0).
    weighted_by : np.ndarray, optional
        Optional 2D weight array (e.g., reflectivity). If provided,
        centroid is weighted by these values.
    
    Returns
    -------
    centroid : tuple of (y, x) or None
        (row, col) coordinates of centroid, or None if label not found.
    """
    if label_id <= 0:
        logger.warning("Centroid requested for background label %d", label_id)
        return None
    
    mask = (label_array == label_id)
    
    if not mask.any():
        return None
    
    if weighted_by is not None:
        # Weighted centroid
        weights = weighted_by.copy()
        weights[~mask] = 0
        
        total_weight = weights.sum()
        if total_weight <= 0:
            # Fall back to unweighted
            return _unweighted_centroid(mask)
        
        y_coords, x_coords = np.where(mask)
        pixel_weights = weights[y_coords, x_coords]
        
        centroid_y = np.average(y_coords, weights=pixel_weights)
        centroid_x = np.average(x_coords, weights=pixel_weights)
        
        return (centroid_y, centroid_x)
    else:
        # Unweighted centroid (simple geometric mean)
        return _unweighted_centroid(mask)


def _unweighted_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """Compute unweighted centroid from binary mask.
    
    Parameters
    ----------
    mask : np.ndarray
        2D binary mask (True where label is present).
    
    Returns
    -------
    centroid : tuple of (y, x)
        (row, col) coordinates.
    """
    y_coords, x_coords = np.where(mask)
    centroid_y = y_coords.mean()
    centroid_x = x_coords.mean()
    return (centroid_y, centroid_x)


def compute_all_cell_centroids(label_array: np.ndarray,
                               weighted_by: np.ndarray = None) -> Dict[int, Tuple[float, float]]:
    """Compute centroids for all labeled cells in an array.
    
    Parameters
    ----------
    label_array : np.ndarray
        2D array of integer labels.
    weighted_by : np.ndarray, optional
        Optional 2D weight array for weighted centroid.
    
    Returns
    -------
    centroids : dict
        Mapping of label_id -> (y_centroid, x_centroid).
        Excludes label 0 (background).
    """
    unique_labels = np.unique(label_array[label_array > 0])
    
    centroids = {}
    for label_id in unique_labels:
        centroid = compute_cell_centroid(label_array, int(label_id), weighted_by=weighted_by)
        if centroid is not None:
            centroids[int(label_id)] = centroid
    
    return centroids


# ============================================================================
# PROJECTION VECTOR COMPUTATION
# ============================================================================

def compute_projection_vector(src_centroid: Tuple[float, float],
                             proj_centroid: Tuple[float, float]) -> Tuple[float, float]:
    """Compute displacement vector from source to projected centroid.
    
    Parameters
    ----------
    src_centroid : tuple of (y, x)
        Source cell centroid.
    proj_centroid : tuple of (y, x)
        Projected cell centroid.
    
    Returns
    -------
    displacement : tuple of (dy, dx)
        Displacement vector.
    """
    dy = proj_centroid[0] - src_centroid[0]
    dx = proj_centroid[1] - src_centroid[1]
    return (dy, dx)


def compute_projected_position(src_centroid: Tuple[float, float],
                               flow_field: np.ndarray) -> Optional[Tuple[float, float]]:
    """Compute projected position using optical flow at source centroid.
    
    Useful for predicting where a cell will be based on flow field.
    
    Parameters
    ----------
    src_centroid : tuple of (y, x)
        Source cell centroid.
    flow_field : np.ndarray
        Optical flow array (H, W, 2).
    
    Returns
    -------
    proj_position : tuple of (y, x) or None
        Projected position, or None if centroid out of bounds.
    """
    y, x = src_centroid
    H, W = flow_field.shape[:2]
    
    # Check bounds
    if not (0 <= y < H and 0 <= x < W):
        logger.warning("Centroid (%.1f, %.1f) out of bounds (H=%d, W=%d)", y, x, H, W)
        return None
    
    # Round to nearest pixel for flow lookup
    yi = int(np.round(y))
    xi = int(np.round(x))
    
    # Clamp to bounds
    yi = np.clip(yi, 0, H - 1)
    xi = np.clip(xi, 0, W - 1)
    
    fx, fy = flow_field[yi, xi]
    
    # Apply flow
    proj_y = y + fy
    proj_x = x + fx
    
    return (proj_y, proj_x)


# ============================================================================
# CELL GEOMETRY OPERATIONS
# ============================================================================

def fill_cell_with_convex_hull(label_array: np.ndarray, label_id: int) -> np.ndarray:
    """Fill a labeled region with its convex hull.
    
    Parameters
    ----------
    label_array : np.ndarray
        2D label array.
    label_id : int
        Label ID to fill.
    
    Returns
    -------
    filled_mask : np.ndarray
        Binary mask of filled region.
    """
    if convex_hull_image is None:
        logger.error("scikit-image not available; cannot compute convex hull")
        return np.zeros_like(label_array, dtype=bool)
    
    mask = (label_array == label_id).astype(np.uint8)
    filled = convex_hull_image(mask)
    
    return filled.astype(bool)


def apply_morphological_closing(label_array: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply morphological closing to all labeled regions.
    
    Closing = dilation followed by erosion. Fills small holes and gaps.
    
    Parameters
    ----------
    label_array : np.ndarray
        2D label array (can have NaN for background).
    kernel_size : int
        Kernel size for morphological operations.
    
    Returns
    -------
    closed_labels : np.ndarray
        Closed label array.
    """
    if binary_closing is None:
        logger.error("scipy.ndimage not available; cannot apply morphological closing")
        return label_array.copy()
    
    closed = label_array.copy()
    unique_labels = np.unique(label_array[~np.isnan(label_array)])
    
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    
    for label_id in unique_labels:
        if label_id <= 0:
            continue  # Skip background
        
        mask = (label_array == label_id).astype(np.uint8)
        closed_mask = binary_closing(mask, structure=kernel)
        
        closed[closed_mask > 0] = label_id
    
    return closed


def extract_cell_boundary(label_array: np.ndarray, label_id: int) -> np.ndarray:
    """Extract boundary pixels of a labeled region.
    
    Parameters
    ----------
    label_array : np.ndarray
        2D label array.
    label_id : int
        Label ID.
    
    Returns
    -------
    boundary : np.ndarray
        Binary mask of boundary pixels.
    """
    mask = (label_array == label_id).astype(np.uint8)
    H, W = mask.shape
    
    boundary = np.zeros_like(mask)
    
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if mask[y, x] == 1:
                # Check if any neighbor differs
                neighbors = [
                    mask[y-1, x-1], mask[y-1, x], mask[y-1, x+1],
                    mask[y, x-1],                   mask[y, x+1],
                    mask[y+1, x-1], mask[y+1, x], mask[y+1, x+1]
                ]
                if any(n != mask[y, x] for n in neighbors):
                    boundary[y, x] = 1
    
    return boundary.astype(bool)


# ============================================================================
# CELL PROPERTY COMPUTATION
# ============================================================================

def compute_cell_area(label_array: np.ndarray, label_id: int,
                     pixel_area_km2: float = 1.0) -> float:
    """Compute area of a labeled cell.
    
    Parameters
    ----------
    label_array : np.ndarray
        2D label array.
    label_id : int
        Label ID.
    pixel_area_km2 : float
        Area of one pixel in km².
    
    Returns
    -------
    area : float
        Cell area in km².
    """
    mask = (label_array == label_id)
    num_pixels = mask.sum()
    return num_pixels * pixel_area_km2


def compute_cell_compactness(label_array: np.ndarray, label_id: int) -> Optional[float]:
    """Compute compactness of a labeled cell (perimeter²/area).
    
    Lower values indicate more circular/compact cells.
    
    Parameters
    ----------
    label_array : np.ndarray
        2D label array.
    label_id : int
        Label ID.
    
    Returns
    -------
    compactness : float or None
        Perimeter² / area, or None if cell has area 0.
    """
    mask = (label_array == label_id).astype(np.uint8)
    area = mask.sum()
    
    if area == 0:
        return None
    
    # Simple boundary-based perimeter estimate
    boundary = extract_cell_boundary(label_array, label_id)
    perimeter = boundary.sum()
    
    if perimeter == 0:
        return None
    
    compactness = (perimeter ** 2) / area
    return compactness


if __name__ == "__main__":
    print("Geometric utilities loaded. Use in cell analysis modules.")
