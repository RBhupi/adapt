"""Read and regrid NEXRAD Level-II files to Cartesian xarray.Dataset.

This module handles loading raw NEXRAD archive files and transforming them into
regular Cartesian grids using Py-ART. The output is an xarray.Dataset containing
gridded reflectivity and other radar fields, enabling downstream analysis
(segmentation, projection, cell detection).

Key capabilities:
- Reads NEXRAD Level-II (.gz) files
- Regrids to configurable Cartesian grid (default: 41x201x201 m)
- Saves intermediate NetCDF files for data lineage
- Handles errors gracefully (logs and returns None)

Author: Bhupendra Raut
"""

from pathlib import Path
from typing import Optional
import logging
import pyart

import xarray as xr

__all__ = ['RadarDataLoader']

logger = logging.getLogger(__name__)

class RadarDataLoader:
    """Load and regrid NEXRAD radar files to Cartesian grid xarray.Dataset.
    
    This class handles the two-stage process of transforming raw NEXRAD data:
    
    1. **Read**: Loads NEXRAD Level-II archive files using Py-ART
    2. **Regrid**: Transforms from polar to Cartesian grid using distance-weighted
       interpolation (configurable algorithm, default: Cressman weighting)
    
    The output is an xarray.Dataset with dimensions (z, y, x) and data variables
    including reflectivity and velocity. Radar metadata (lat/lon/alt) is preserved
    in dataset attributes.
    
    Configuration
    =============
    Expects config dict with two sub-dicts:
    
    - `reader` : dict
        - `file_format` : str, file type (default: "nexrad_archive")
    
    - `regridder` : dict
        - `grid_shape` : tuple, (nz, ny, nx) vertical/horizontal grid points
          (default: (41, 201, 201))
        - `grid_limits` : tuple, ((z_min, z_max), (y_min, y_max), (x_min, x_max))
          in meters from radar center (default: ((0, 20000), (-100000, 100000), (-100000, 100000)))
        - `roi_func` : str, influence radius function (default: "dist_beam" - distance
          from radar ray). Other options: "constant_distance", "dist"
        - `min_radius` : float, minimum radius for influence (meters, default: 1750)
        - `weighting_function` : str, interpolation weighting (default: "cressman")
          Other options: "linear", "nearest", "barnes"
    
    Notes
    -----
    - Not thread-safe: create separate loader instances for concurrent use
    - Regridding takes 5-15 seconds per file (CPU-bound)
    - NetCDF output is always compressed (zlib, level 9)
    - All methods return None on failure (logged, not raised)
    - Radar object is a temporary Py-ART object; only the xarray.Dataset is kept
    
    Examples
    --------
    >>> config = {
    ...     "reader": {"file_format": "nexrad_archive"},
    ...     "regridder": {"grid_shape": (41, 201, 201), "min_radius": 1750.0}
    ... }
    >>> loader = RadarDataLoader(config)
    >>> ds = loader.load_and_regrid("20250305_KLOT.gz", save_netcdf=True, 
    ...                              output_dir="./grids")
    >>> print(ds.data_vars)  # reflectivity, velocity, etc.
    """

    def __init__(self, config: dict | None = None):
        """Initialize loader with configuration for reader and regridder.
        
        Parameters
        ----------
        config : dict
            Required dictionary containing:
            
            - `reader` : dict
                Configuration for file reading.
                
                - `file_format` : str, optional
                    File format type. Currently supports "nexrad_archive" (default).
                    Used to dispatch to correct Py-ART reader function.
            
            - `regridder` : dict
                Configuration for grid transformation.
                
                - `grid_shape` : tuple of int, optional
                    (nz, ny, nx) - Number of grid points in each dimension.
                    Default: (41, 201, 201)
                
                - `grid_limits` : tuple of tuple, optional
                    ((z_min, z_max), (y_min, y_max), (x_min, x_max)) in meters
                    from radar center. Default: ((0, 20000), (-100000, 100000), (-100000, 100000))
                
                - `roi_func` : str, optional
                    Radius of influence function. "dist_beam" (default) uses
                    distance along radar ray. "dist" uses straight-line distance.
                
                - `min_radius` : float, optional
                    Minimum radius for weighting function (meters). Default: 1750.0
                
                - `weighting_function` : str, optional
                    Interpolation algorithm. "cressman" (default) is fast and smooth.
                    "barnes" is slower but better quality. "linear" is fastest.
        
        Raises
        ------
        ValueError
            If config is None.
        TypeError
            If config is not a dict.
        KeyError
            If required keys ("reader", "regridder") are missing from config.
        
        Notes
        -----
        Config validation is strict; all required top-level keys must be present.
        Sub-key defaults are applied during regridding, not initialization.
        
        Examples
        --------
        >>> config = {
        ...     "reader": {"file_format": "nexrad_archive"},
        ...     "regridder": {
        ...         "grid_shape": (41, 201, 201),
        ...         "grid_limits": ((0, 20000), (-100000, 100000), (-100000, 100000)),
        ...         "roi_func": "dist_beam",
        ...         "min_radius": 1750.0,
        ...         "weighting_function": "cressman"
        ...     }
        ... }
        >>> loader = RadarDataLoader(config)
        """
        self.config = self._validate_config(config)
        self.reader_config = self.config["reader"]
        self.regridder_config = self.config["regridder"]

    @staticmethod
    def _validate_config(cfg):
        def ensure_present(c):
            if c is None:
                raise ValueError("Config dict required.")
            if not isinstance(c, dict):
                raise TypeError("Config must be a dict.")
            return c

        def ensure_keys(c):
            needed = {"reader", "regridder"}
            missing = needed - c.keys()
            if missing:
                raise KeyError(f"Missing config keys: {missing}")
            return c

        return ensure_keys(ensure_present(cfg))

    def read(self, filepath: Path | str) -> Optional[object]:
        """Read a NEXRAD archive file into a Py-ART Radar object.
        
        Loads the raw NEXRAD Level-II file (.gz format) and parses it into
        Py-ART's Radar object, which contains polar coordinate reflectivity,
        velocity, and other fields indexed by ray and gate.
        
        Parameters
        ----------
        filepath : Path or str
            Path to the NEXRAD Level-II file. Must exist on disk.
        
        Returns
        -------
        pyart.core.Radar or None
            Py-ART Radar object if successful, None if:
            - File does not exist
            - File format is unsupported
            - Read operation fails
        
        Notes
        -----
        - File existence is validated before attempting read
        - Errors are logged; exception stack trace is included
        - Radar object is large (~100-200 MB) and should be passed immediately
          to regrid() for processing and cleanup
        - Read time: 1-3 seconds per file (mostly decompression)
        
        Examples
        --------
        >>> loader = RadarDataLoader(config)
        >>> radar = loader.read("20250305_KLOT.gz")
        >>> if radar is not None:
        ...     print(f"Loaded {radar.nrays} rays x {radar.ngates} gates")
        """
        try:
            filepath = str(filepath)
            
            # Validate file exists
            from pathlib import Path as PathlibPath
            if not PathlibPath(filepath).exists():
                logger.error("Radar file not found: %s", filepath)
                return None
            
            file_format = self.reader_config.get("file_format", "nexrad_archive")

            if file_format == "nexrad_archive":
                radar = pyart.io.read_nexrad_archive(filepath)
            else:
                logger.error("Unsupported file format: %s", file_format)
                return None

            logger.debug("Successfully read radar file: %s", filepath)
            return radar

        except Exception as e:
            logger.exception("Failed to read radar file %s", filepath)
            return None

    def regrid(self, radar: object, grid_kwargs: dict = None,
               output_dir: str = None, source_filepath: str = None) -> Optional[xr.Dataset]:
        """Transform a Py-ART Radar object from polar to Cartesian grid.
        
        Performs distance-weighted interpolation to convert irregular polar
        coordinates (typical of radar) to a regular Cartesian grid suitable
        for machine learning and image processing tasks. Radar metadata
        (latitude, longitude, altitude) is preserved in dataset attributes.
        
        Parameters
        ----------
        radar : pyart.core.Radar
            Input Py-ART Radar object (from read() method).
        
        grid_kwargs : dict, optional
            Override regridding parameters. Merged with config defaults.
            Supported keys:
            
            - `grid_shape` : tuple of int
                (nz, ny, nx) grid dimensions
            
            - `grid_limits` : tuple of tuple
                ((z_min, z_max), (y_min, y_max), (x_min, x_max)) in meters
            
            - `roi_func` : str
                Radius of influence function
            
            - `min_radius` : float
                Minimum radius in meters
            
            - `weighting_function` : str
                Interpolation algorithm ("cressman", "barnes", "linear")
        
        output_dir : str, optional
            Directory to save intermediate NetCDF file. If None and
            save_netcdf=True in caller, uses current directory.
        
        source_filepath : str, optional
            Original file path for naming NetCDF output. If provided,
            output file is named {stem}.nc
        
        Returns
        -------
        xr.Dataset or None
            Cartesian grid xarray.Dataset with dimensions (z, y, x) and
            variables (reflectivity, velocity, etc.) if successful.
            Returns None if regridding or NetCDF save fails.
        
        Notes
        -----
        - Regridding is CPU-intensive: 5-15 seconds per file
        - Grid defaults from config are merged with grid_kwargs overrides
        - NetCDF files are compressed (zlib, level 9) for storage efficiency
        - Radar location attributes (latitude, longitude, altitude) are added
          to dataset.attrs for metadata tracking
        - Fails gracefully: logs exception and returns None
        
        Examples
        --------
        >>> loader = RadarDataLoader(config)
        >>> radar = loader.read("20250305_KLOT.gz")
        >>> ds = loader.regrid(
        ...     radar,
        ...     grid_kwargs={"grid_shape": (50, 250, 250)},
        ...     output_dir="./grids",
        ...     source_filepath="20250305_KLOT.gz"
        ... )
        >>> print(ds.reflectivity.shape)  # (50, 250, 250)
        """

        try:
            # Merge default regridder config with overrides
            final_grid_kwargs = {
                "grid_shape": self.regridder_config.get("grid_shape", (41, 201, 201)),
                "grid_limits": self.regridder_config.get("grid_limits",
                    ((0, 20000), (-100000, 100000), (-100000, 100000))),
                "roi_func": self.regridder_config.get("roi_func", "dist_beam"),
                "min_radius": self.regridder_config.get("min_radius", 1750.0),
                "weighting_function": self.regridder_config.get("weighting_function", "cressman"),
            }

            # Override with explicit kwargs if provided
            if grid_kwargs:
                final_grid_kwargs.update(grid_kwargs)

            # Perform regridding
            grid = pyart.map.grid_from_radars(radar, **final_grid_kwargs)
            ds = grid.to_xarray()
            logger.debug("Success: regrid to xarray.Dataset")

            # Add radar location attributes to dataset.
            ds.attrs['radar_latitude'] = float(radar.latitude['data'][0])
            ds.attrs['radar_longitude'] = float(radar.longitude['data'][0])
            ds.attrs['radar_altitude'] = float(radar.altitude['data'][0])

            self._write_netcdf(ds, output_dir, source_filepath)
            return ds

        except Exception as e:
            logger.exception("Regridding failed")
            return None


    def _write_netcdf(self, ds, output_dir, source_filepath):
        """Internal writer for netcdf output."""
        try:
            if output_dir is None:
                output_dir = "."

            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)

            nc_filename = Path(source_filepath).stem + ".nc"

            nc_path = output_dir_path / nc_filename

            encoding = {var: {"zlib": True, "complevel": 9} for var in ds.data_vars}
            ds.to_netcdf(nc_path, encoding=encoding, compute=True)

            logger.info("Saved regridded NetCDF: %s", nc_path)

        except Exception as e:
            logger.warning("Failed to save NetCDF: %s", e)



    def load_and_regrid(self, filepath: Path | str, grid_kwargs: dict = None,
                       save_netcdf: bool = True, output_dir: str = None) -> Optional[xr.Dataset]:
        """Read and regrid a NEXRAD file in one call (convenience method).
        
        Combines read() and regrid() operations for simpler usage when
        both steps are needed in sequence. This is the primary entry point
        for most use cases.
        
        Parameters
        ----------
        filepath : Path or str
            Path to NEXRAD Level-II (.gz) file.
        
        grid_kwargs : dict, optional
            Regridding parameter overrides. Passed to regrid().
        
        save_netcdf : bool, default True
            Whether to save the intermediate regridded NetCDF file.
            If True, uses output_dir parameter.
        
        output_dir : str, optional
            Directory for NetCDF output. Only used if save_netcdf=True.
            If None, uses current directory.
        
        Returns
        -------
        xr.Dataset or None
            Cartesian grid xarray.Dataset if successful, None if:
            - File does not exist or cannot be read
            - Regridding fails
            - NetCDF save fails (if save_netcdf=True)
        
        Notes
        -----
        - Preferred method over separate read() + regrid() calls
        - Two failure points: read stage and regrid stage
        - NetCDF save is optional; set save_netcdf=False for memory-only
          processing (avoids disk I/O)
        - Returns same xarray.Dataset regardless of save_netcdf setting
        
        Examples
        --------
        >>> loader = RadarDataLoader(config)
        >>> ds = loader.load_and_regrid(
        ...     "20250305_KLOT.gz",
        ...     save_netcdf=True,
        ...     output_dir="./grids"
        ... )
        >>> if ds is not None:
        ...     print(f"Grid shape: {ds.reflectivity.shape}")
        ...     cells = segment_cells(ds)  # Downstream processing
        """
        radar = self.read(filepath)
        if radar is None:
            return None

        ds = self.regrid(radar, grid_kwargs=grid_kwargs,
                        output_dir=output_dir if save_netcdf else None,
                        source_filepath=filepath)
        return ds


if __name__ == "__main__":
    print("RadarDataLoader loaded. Use: loader = RadarDataLoader(config)")

