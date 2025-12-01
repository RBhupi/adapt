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


def compute_all_cell_centroids(label_array: np.ndarray, weighted_by: np.ndarray = None) -> Dict[int, Tuple[float, float]]:
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