"""Unified radar data loader: read + regrid to canonical xarray.Dataset.

Reads raw radar files (NEXRAD Level-II) and regrids them to a Cartesian
xarray.Dataset using Py-ART.

Author: Bhupendra Raut
"""

from pathlib import Path
from typing import Optional
import logging
import pyart

import xarray as xr

logger = logging.getLogger(__name__)

class RadarDataLoader:
    """reading and regridding radar data files
    """

    def __init__(self, config: dict | None = None):
        """Init with config checks; expect {reader, regridder}.
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
        """Read a radar file to Py-ART.
        
        Validates file exists before attempting to read.
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
        """ Regrid a Py-ART Radar object to xarray.Dataset.
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
        """read then regrid in one call.
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

