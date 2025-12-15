"""`Adapt` Expert Configuration.

Advanced pipeline settings for optical flow, visualization, and algorithms.
Most users should only modify scripts/user_config.py

This file provides default configuration values and algorithmic parameters.
The run_pipeline.py script merges user config with these expert settings.

Author: Bhupendra Raut
"""

from pathlib import Path
from typing import Dict, Tuple

# Default pipeline configuration
PARAM_CONFIG: Dict = {
    "mode": "realtime",
    
    "reader": {
        "file_format": "nexrad_archive",
    },
    
    "downloader": {
        "radar_id": None,
        "output_dir": None,
        "latest_n": 5,
        "minutes": 60,
        "sleep_interval": 300,
        "start_time": None,
        "end_time": None,
    },
    
    "regridder": {
        "grid_shape": (41, 301, 301),
        "grid_limits": ((0, 20000), (-150000, 150000), (-150000, 150000)),
        "roi_func": "dist_beam",
        "min_radius": 1750.0,
        "weighting_function": "cressman",
        "save_netcdf": True,
    },
    
    "segmenter": {
        "method": "threshold",
        "threshold": 30,
        "min_cellsize_gridpoint": 5,
        "max_cellsize_gridpoint": None,
        "closing_kernel": (2, 2),
        "filter_by_size": True,
    },
    
    "global": {
        "z_level": 2000,
        "var_names": {
            "reflectivity": "reflectivity",
            "cell_labels": "cell_labels",
        },
        "coord_names": {
            "time": "time",
            "z": "z",
            "y": "y",
            "x": "x",
        },
    },
    
    "projector": {
        "method": "adapt_default",
        "max_time_interval_minutes": 30,
        "max_projection_steps": 5,
        "nan_fill_value": 0,
        "flow_params": {
            "pyr_scale": 0.5,
            "levels": 3,  # Pyramid levels
            "winsize": 21,  # Window size
            "iterations": 10,  # Number of iterations
            "poly_n": 7,  # Polynomial expansion degree
            "poly_sigma": 1.5,  # Gaussian sigma
            "flags": 0,
        },
        "min_motion_threshold": 0.5,  # Minimum motion threshold
    },
    
    "analyzer": {
        # Whitelist of radar measurement variables to analyze
        # Only these variables will have statistics extracted and saved to database
        "radar_variables": [
            "reflectivity",
            "velocity",
            "differential_phase",
            "differential_reflectivity",
            "spectrum_width",
            "cross_correlation_ratio",
        ],
        # Fields to always exclude (metadata, segmentation, flow vectors)
        "exclude_fields": [
            "ROI", "labels", "cell_labels",  # Segmentation metadata
            "cell_projections",  # Projection metadata
            "clutter_filter_power_removed"  # Clutter filter
        ],
    },
    
    "visualization": {
        "enabled": True,
        "dpi": 200,
        "figsize": (18, 8),  # Width x Height in inches
        "output_format": "png",  # png, pdf, or jpeg
        "use_basemap": True,
        "basemap_alpha": 0.6,
        
        # Contour styling
        "seg_linewidth": 0.8,  # Segmentation contour width
        "proj_linewidth": 1.0,  # Projection contour width
        "proj_alpha": 0.8,  # Projection contour transparency
        
        # Flow vector styling
        "flow_scale": 0.5,  # Lower = larger arrows (15-40 for long arrows)
        "flow_subsample": 10,  # Sample every Nth pixel
        
        # Reflectivity thresholds
        "min_reflectivity": 10,
        "refl_vmin": 10,
        "refl_vmax": 50,
    },
    
    "output": {
        "compression": "snappy",
    },
    
    "logging": {
        "level": "INFO",
    },
}


def get_grid_kwargs(config: Dict = None) -> Dict:
    """Get grid kwargs for PyART regridding.
    
    Parameters
    ----------
    config : dict, optional
        Pipeline configuration with 'regridder' section.
    
    Returns
    -------
    dict
        Grid kwargs for pyart.map.grid_from_radars()
    """
    config = config or PARAM_CONFIG
    regridder = config.get("regridder", {})
    
    return {
        "grid_shape": regridder.get("grid_shape", (41, 301, 301)),
        "grid_limits": regridder.get("grid_limits", ((0, 20000), (-150000, 150000), (-150000, 150000))),
        "roi_func": regridder.get("roi_func", "dist_beam"),
        "min_radius": regridder.get("min_radius", 1750.0),
        "weighting_function": regridder.get("weighting_function", "cressman"),
    }


def get_output_path(config: Dict = None) -> Path:
    """Get default output path for SQLite database.
    
    Parameters
    ----------
    config : dict, optional
        Pipeline configuration.
    
    Returns
    -------
    Path
        Output file path for SQLite database.
    """
    config = config or PARAM_CONFIG
    
    output_dirs = config.get("output_dirs", {})
    if output_dirs:
        analysis_dir = Path(output_dirs.get("analysis", "."))
        radar_id = config.get("downloader", {}).get("radar_id", "UNKNOWN")
        return analysis_dir / f"{radar_id}_cells_statistics.db"
    
    # Fallback
    return Path(".") / "cells_statistics.db"
