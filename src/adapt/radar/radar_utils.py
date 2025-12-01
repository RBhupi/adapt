#!/usr/bin/env python3
"""Radar utilities for cell analysis and projection.

Centralized utilities for:
- Grid spacing and time interval computation
- Cell centroids and geometry
- Projection vectors

@TODO: Add more common utilities from other modules here.
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
    # ...existing code or placeholder...
    pass


def compute_cell_centroid(label_array: np.ndarray, label_id: int, weighted_by: np.ndarray = None) -> Optional[Tuple[float, float]]:
    """
    Compute the centroid (y, x) of a single labeled cell.
    If weighted_by is provided, compute the weighted centroid.
    """
    mask = (label_array == label_id)
    if not np.any(mask):
        return None
    if weighted_by is not None:
        centroid = center_of_mass(weighted_by, labels=mask, index=True)
    else:
        centroid = center_of_mass(mask)
    return centroid

def compute_all_cell_centroids(label_array: np.ndarray, weighted_by: np.ndarray = None) -> Dict[int, Tuple[float, float]]:
    """
    Compute centroids for all labeled cells in an array.

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