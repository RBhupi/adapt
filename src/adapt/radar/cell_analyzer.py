# src/adapt/radar/cell_analyzer.py
"""Radar cell property analyzer.

Extracts statistics from labeled cells for database storage.

Author: Bhupendra Raut
"""

import logging
import json
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import center_of_mass
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
    - cell_projections: projection centroids (offset, y, x)
    - reflectivity and other fields
    """
    
    def __init__(self, config: dict = None):
        """Initialize analyzer with config.
        
        Parameters
        ----------
        config : dict
            Configuration with 'global' section for var_names/coord_names,
            'radar_variables' list for variables to analyze,
            'exclude_fields' list for variables to skip,
            and 'projector' section with max_projection_steps.
        """
        self.config = config or {}
        self._global_config = self.config.get("global", {})
        self._projector_config = self.config.get("projector", {})
        
        # Get variable names from config
        var_names = self._global_config.get("var_names", {})
        self.reflectivity_field = var_names.get("reflectivity", "reflectivity")
        
        # Whitelist of radar variables to analyze (only these get stats computed)
        self.radar_variables = self.config.get(
            "radar_variables",
            [
                "reflectivity", "velocity", "differential_phase",
                "differential_reflectivity", "spectrum_width",
                "cross_correlation_ratio",
            ]
        )
        
        # Fields to exclude from statistics (metadata, flow vectors, etc.)
        self.exclude_fields = self.config.get(
            "exclude_fields", 
            [
                "ROI", "labels", "cell_labels",  # Segmentation metadata
                "clutter_filter_power_removed",  # Clutter filter
                "cell_projections",  # Projection metadata
            ]
        )
        
        # Get max projection steps from config
        self.max_projection_steps = self._projector_config.get("max_projection_steps", 5)

    def extract(self, ds: xr.Dataset, z_level: int = None) -> pd.DataFrame:
        """Extract centroid and statistics from labeled cells.

        Parameters
        ----------
        ds : xr.Dataset
            2D segmented dataset with cell_labels, reflectivity, and optional flow fields.
        z_level : int, optional
            Unused; kept for API compatibility.

        Returns
        -------
        pd.DataFrame
            Cell properties: cell_label, centroids (geom/mass/maxdbz with x/y/lat/lon), area, radar stats.
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
        """Get list of radar variables suitable for statistics analysis.
        
        Uses whitelist approach: only variables in radar_variables config
        that are actually present in the dataset are analyzed.
        """
        available_vars = []
        for var in self.radar_variables:
            if var in ds.data_vars and var not in self.exclude_fields:
                # Check if it's 2D (y, x) or 3D (z, y, x)
                if ds[var].dims[-2:] == ("y", "x") or ds[var].dims[-3:] == ("z", "y", "x"):
                    available_vars.append(var)
        return available_vars

    def _compute_geometric_centroid(self, mask, lat_grid=None, lon_grid=None):
        """Compute geometric centroid (center of mass) of cell region.
        
        Parameters
        ----------
        mask : np.ndarray
            Boolean mask of cell region
        lat_grid : np.ndarray, optional
            Latitude grid for geographic coordinates
        lon_grid : np.ndarray, optional
            Longitude grid for geographic coordinates
            
        Returns
        -------
        dict
            Centroid coordinates: centroid_x, centroid_y, centroid_lat, centroid_lon
        """
        centroid_y, centroid_x = center_of_mass(mask.astype(float))
        centroid_x = int(np.round(centroid_x))
        centroid_y = int(np.round(centroid_y))
        
        result = {
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
        }
        
        if lat_grid is not None and lon_grid is not None:
            lat, lon = self.get_lat_lon(centroid_x, centroid_y, lat_grid, lon_grid)
            result["centroid_lat"] = float(lat)
            result["centroid_lon"] = float(lon)
        
        return result

    def _extract_field_values(self, ds, var, mask):
        """Extract field values at mask locations for 2D data."""
        data = ds[var].values
        return data[mask]

    def _extract_region_props(self, region, label_array, refl, lat_grid, lon_grid,
                              ds, data_vars, pixel_area_km2):
        """Extract properties for a single cell region.
        
        Naming convention - ALL centroids stored in both XY and lat/lon:
        - cell_centroid_<type>_x, cell_centroid_<type>_y: Pixel coordinates
        - cell_centroid_<type>_lat, cell_centroid_<type>_lon: Geographic coordinates
        
        Centroid types:
        - geom: Geometric centroid (center of mass of binary mask)
        - mass: Mass-weighted centroid (reflectivity weighted)
        - maxdbz: Maximum reflectivity centroid
        - registration_<idx>: Registration/projection centroids (index 0 = registration)
        - projection_<idx>: Forward projection centroids (indices 1+)
        
        Other naming:
        - cell_heading_<stat>: Heading vector statistics within cell
        - radar_<variable>_<stat>: Radar variable statistics
        """
        mask = label_array == region.label
        region_coords = region.coords
        region_values = refl[tuple(region_coords.T)]
        max_idx = np.argmax(region_values)
        max_coord = region_coords[max_idx]

        # Get scan start time
        scan_time = str(ds.time.values) if "time" in ds.coords else ""

        # === GEOMETRIC CENTROID (center of mass of binary mask) ===
        geom_props = self._compute_geometric_centroid(mask, lat_grid, lon_grid)
        
        # === MAX REFLECTIVITY CENTROID ===
        centroid_maxdbz_y = int(np.round(max_coord[0]))
        centroid_maxdbz_x = int(np.round(max_coord[1]))
        lat_maxdbz = float(lat_grid[tuple(max_coord)])
        lon_maxdbz = float(lon_grid[tuple(max_coord)])

        # === MASS-WEIGHTED CENTROID (reflectivity weighted) ===
        refl_cell = refl[mask]
        if len(refl_cell) > 0 and np.any(np.isfinite(refl_cell)):
            y_indices, x_indices = np.where(mask)
            valid_mask = np.isfinite(refl_cell)
            if np.any(valid_mask):
                centroid_mass_y = int(np.round(np.average(y_indices[valid_mask], weights=refl_cell[valid_mask])))
                centroid_mass_x = int(np.round(np.average(x_indices[valid_mask], weights=refl_cell[valid_mask])))
            else:
                centroid_mass_y, centroid_mass_x = int(np.round(geom_props["centroid_y"])), int(np.round(geom_props["centroid_x"]))
        else:
            centroid_mass_y, centroid_mass_x = int(np.round(geom_props["centroid_y"])), int(np.round(geom_props["centroid_x"]))

        lat_mass, lon_mass = self.get_lat_lon(centroid_mass_x, centroid_mass_y, lat_grid, lon_grid)

        # Build properties dict with ALL centroids in both XY and lat/lon
        props = {
            "time_volume_start": scan_time,  # Start of radar volume scan
            "time": scan_time,  # Analysis time (same as time_volume_start)
            "time_volume_end": None,  # Will be populated when available
            "cell_label": int(region.label),
            "cell_area_sqkm": float(region.area * pixel_area_km2),
            # Geometric centroid - both XY and lat/lon
            "cell_centroid_geom_x": geom_props["centroid_x"],
            "cell_centroid_geom_y": geom_props["centroid_y"],
            "cell_centroid_geom_lat": geom_props.get("centroid_lat", np.nan),
            "cell_centroid_geom_lon": geom_props.get("centroid_lon", np.nan),
            # Max reflectivity centroid - both XY and lat/lon
            "cell_centroid_maxdbz_x": centroid_maxdbz_x,
            "cell_centroid_maxdbz_y": centroid_maxdbz_y,
            "cell_centroid_maxdbz_lat": lat_maxdbz,
            "cell_centroid_maxdbz_lon": lon_maxdbz,
            # Mass-weighted centroid - both XY and lat/lon
            "cell_centroid_mass_x": centroid_mass_x,
            "cell_centroid_mass_y": centroid_mass_y,
            "cell_centroid_mass_lat": float(lat_mass),
            "cell_centroid_mass_lon": float(lon_mass),
        }

        # === HEADING VECTOR STATISTICS ===
        if "heading_x" in ds.data_vars and "heading_y" in ds.data_vars:
            try:
                heading_x_vals = self._extract_field_values(ds, "heading_x", mask)
                heading_y_vals = self._extract_field_values(ds, "heading_y", mask)
                if heading_x_vals.size > 0 and heading_y_vals.size > 0:
                    props["cell_heading_x_mean"] = float(np.nanmean(heading_x_vals))
                    props["cell_heading_y_mean"] = float(np.nanmean(heading_y_vals))
            except Exception as e:
                logger.debug("Could not extract heading vectors: %s", e)

        # === PROJECTION CENTROIDS (registration + forward projections) ===
        # Store ALL in both XY and lat/lon coordinates
        if "cell_projections" in ds.data_vars:
            try:
                projections = ds["cell_projections"].values
                if projections.ndim == 3:  # (offset, y, x)
                    projection_centroids = []
                    
                    # Extract centroids for each projection step
                    for step_idx in range(min(projections.shape[0], self.max_projection_steps + 1)):
                        proj_mask = projections[step_idx] == region.label
                        if np.any(proj_mask):
                            # Use reusable centroid function (already has lat/lon)
                            proj_centroid = self._compute_geometric_centroid(proj_mask, lat_grid, lon_grid)
                            projection_centroids.append(proj_centroid)
                        else:
                            projection_centroids.append(None)
                    
                    # Store each centroid in both XY and lat/lon
                    if projection_centroids:
                        # Index 0 = Registration centroid (projection from previous to current frame)
                        if projection_centroids[0] is not None:
                            reg_cent = projection_centroids[0]
                            props["cell_centroid_registration_x"] = reg_cent["centroid_x"]
                            props["cell_centroid_registration_y"] = reg_cent["centroid_y"]
                            if "centroid_lat" in reg_cent:
                                props["cell_centroid_registration_lat"] = reg_cent["centroid_lat"]
                                props["cell_centroid_registration_lon"] = reg_cent["centroid_lon"]
                        
                        # Indices 1+ = Forward projection centroids
                        for proj_idx, proj_cent in enumerate(projection_centroids[1:], start=1):
                            if proj_cent is not None:
                                props[f"cell_centroid_projection{proj_idx}_x"] = proj_cent["centroid_x"]
                                props[f"cell_centroid_projection{proj_idx}_y"] = proj_cent["centroid_y"]
                                if "centroid_lat" in proj_cent:
                                    props[f"cell_centroid_projection{proj_idx}_lat"] = proj_cent["centroid_lat"]
                                    props[f"cell_centroid_projection{proj_idx}_lon"] = proj_cent["centroid_lon"]
                        
                        # Also store full projection centroids as JSON for compact storage
                        props["cell_projection_centroids_json"] = json.dumps([
                            {k: v for k, v in c.items() if c and not (isinstance(v, float) and np.isnan(v))} if c else None
                            for c in projection_centroids
                        ])
            except Exception as e:
                logger.debug("Could not extract projection centroids: %s", e)

        # Add statistics for each valid data variable (2D) with radar_ prefix
        for var in data_vars:
            try:
                vals = self._extract_field_values(ds, var, mask)
                if vals.size > 0:
                    props[f"radar_{var}_mean"] = float(np.nanmean(vals))
                    props[f"radar_{var}_min"] = float(np.nanmin(vals))
                    props[f"radar_{var}_max"] = float(np.nanmax(vals))
            except Exception as e:
                logger.warning("Skipped var '%s': %s", var, e)

        return props

    @staticmethod
    def get_lat_lon(ix, iy, lat_grid, lon_grid):
        """Extract lat–lon coordinates using integer pixel indices.
        
        Parameters
        ----------
        ix : int
            Integer x (column) index
        iy : int
            Integer y (row) index
        lat_grid : np.ndarray
            2D latitude array [y, x]
        lon_grid : np.ndarray
            2D longitude array [y, x]
            
        Returns
        -------
        tuple
            (lat, lon) as floats, or (np.nan, np.nan) if out of bounds or invalid
        """
        H, W = lat_grid.shape
        
        # Validate bounds
        if not (0 <= ix < W and 0 <= iy < H):
            return np.nan, np.nan
        
        # Extract lat/lon
        lat = lat_grid[iy, ix]
        lon = lon_grid[iy, ix]
        
        # Check for masked or fill values
        if not (np.isfinite(lat) and np.isfinite(lon)):
            return np.nan, np.nan
        
        return float(lat), float(lon)

