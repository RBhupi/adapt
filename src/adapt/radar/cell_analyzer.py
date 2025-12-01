# src/adapt/radar/cell_analyzer.py
"""Radar cell property analyzer.

Extracts statistics from labeled cells for database storage.

Author: Bhupendra Raut
"""

import logging
import numpy as np
import pandas as pd
import xarray as xr
from skimage.measure import regionprops

logger = logging.getLogger(__name__)

# Suppress HDF5 diagnostic error messages
try:
    import h5py
    h5py._errors.silence_errors()
except (ImportError, AttributeError):
    pass


class RadarCellAnalyzer:
    """Extract cell properties from segmented radar data.
    
    Called after projection with ds containing:
    - cell_labels: segmentation labels (y, x)
    - heading_x, heading_y: motion vectors (y, x)
    - reflectivity and other fields
    """
    
    def __init__(self, config: dict = None):
        """Initialize analyzer with config.
        
        Parameters
        ----------
        config : dict
            Configuration with 'global' section for var_names/coord_names,
            and 'exclude_fields' list for variables to skip.
        """
        self.config = config or {}
        self._global_config = self.config.get("global", {})
        
        # Get variable names from config
        var_names = self._global_config.get("var_names", {})
        self.reflectivity_field = var_names.get("reflectivity", "reflectivity")
        
        # Fields to exclude from statistics
        self.exclude_fields = self.config.get(
            "exclude_fields", 
            ["ROI", "labels", "cell_labels", "heading_x", "heading_y"]
        )

    def extract(self, ds: xr.Dataset, z_level: int = None) -> pd.DataFrame:
        """Extract cell properties from a labeled 2D dataset.

        Parameters
        ----------
        ds : xr.Dataset
            2D dataset (already sliced at z-level) with cell_labels from segmenter 
            and optionally heading_x/heading_y from projector.
        z_level : int, optional
            Analysis altitude in meters (not used, kept for compatibility).

        Returns
        -------
        pd.DataFrame
            Cell properties with columns: cell_id, area_km2, lat/lon, field statistics.
        """
        # Get settings from config
        global_cfg = self._global_config
        var_names = global_cfg.get("var_names", {})

        labels_name = var_names.get("cell_labels", "cell_labels")

        # Check for required labels
        if labels_name not in ds.data_vars:
            raise ValueError(
                f"Dataset does not contain '{labels_name}'. "
                "Run the segmenter first."
            )

        # Extract reflectivity (already 2D)
        refl = ds[self.reflectivity_field].values
        label_array = ds[labels_name].values
        pixel_area_km2 = self._pixel_area_km2(ds)

        # Get lat/lon grids
        lat_grid, lon_grid = self._get_lat_lon_grids(ds)
        data_vars = self._get_valid_data_vars(ds)

        # Extract properties for each cell
        results = []
        for region in regionprops(label_array.astype(np.int32), intensity_image=refl):
            if region.label == 0:
                continue

            props = self._extract_region_props(
                region, label_array, refl, lat_grid, lon_grid,
                ds, data_vars, pixel_area_km2
            )
            results.append(props)

        return pd.DataFrame(results)

    def _find_nearest_z(self, ds, z_level, z_name="z"):
        """Find index of nearest z-level."""
        if z_name not in ds.coords:
            return 0
        return int(np.argmin(np.abs(ds[z_name].values - z_level)))

    def _pixel_area_km2(self, ds):
        """Compute pixel area in km²."""
        dx = float(np.abs(ds.x[1] - ds.x[0]))
        dy = float(np.abs(ds.y[1] - ds.y[0]))
        return (dx * dy) / 1e6

    def _get_lat_lon_grids(self, ds):
        """Get lat/lon grids from dataset, generating if needed."""
        if "lat" in ds.coords and "lon" in ds.coords:
            return ds["lat"].values, ds["lon"].values
        elif "lat" in ds.data_vars and "lon" in ds.data_vars:
            return ds["lat"].values, ds["lon"].values
        else:
            # Fallback to origin coordinates
            logger.debug("lat/lon grids not found, using origin coordinates")
            origin_lat = float(ds.attrs.get("origin_latitude", 0.0))
            origin_lon = float(ds.attrs.get("origin_longitude", 0.0))
            
            # Try to get from data variables if not in attrs
            if origin_lat == 0.0 and "origin_latitude" in ds:
                origin_lat = float(ds["origin_latitude"].values)
            if origin_lon == 0.0 and "origin_longitude" in ds:
                origin_lon = float(ds["origin_longitude"].values)

            lat_grid = np.full((len(ds.y), len(ds.x)), origin_lat)
            lon_grid = np.full((len(ds.y), len(ds.x)), origin_lon)
            return lat_grid, lon_grid

    def _get_valid_data_vars(self, ds):
        """Get list of variables suitable for statistics."""
        return [
            v for v in ds.data_vars
            if v not in self.exclude_fields and (
                ds[v].dims[-2:] == ("y", "x") or ds[v].dims[-3:] == ("z", "y", "x")
            )
        ]

    def _extract_field_values(self, ds, var, mask):
        """Extract field values at mask locations for 2D data."""
        data = ds[var].values
        return data[mask]

    def _extract_region_props(self, region, label_array, refl, lat_grid, lon_grid,
                              ds, data_vars, pixel_area_km2):
        """Extract properties for a single cell region (2D data)."""
        mask = label_array == region.label
        region_coords = region.coords
        region_values = refl[tuple(region_coords.T)]
        max_idx = np.argmax(region_values)
        max_coord = region_coords[max_idx]

        lat_max = float(lat_grid[tuple(max_coord)])
        lon_max = float(lon_grid[tuple(max_coord)])

        props = {
            "cell_id": int(region.label),
            "area_km2": float(region.area * pixel_area_km2),
            "lat_max_reflectivity": lat_max,
            "lon_max_reflectivity": lon_max,
        }

        # Add statistics for each valid data variable (2D)
        for var in data_vars:
            try:
                vals = self._extract_field_values(ds, var, mask)
                if vals.size > 0:
                    props[f"{var}_mean"] = float(np.nanmean(vals))
                    props[f"{var}_min"] = float(np.nanmin(vals))
                    props[f"{var}_max"] = float(np.nanmax(vals))
            except Exception as e:
                logger.warning("Skipped var '%s': %s", var, e)

        return props

