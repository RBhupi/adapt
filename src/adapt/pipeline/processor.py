#!/usr/bin/env python3
"""Radar Data Processor: segmentation, and projection and analysis

Receives radar filepaths from queue, processes through the pipeline:
read → regrid → segment →  project → analyze → save

@TODO: break process_file into smaller hierarchical functions.

Author: Bhupendra Raut
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List
import threading
import queue
from datetime import datetime


import pandas as pd
import numpy as np
import xarray as xr
import sqlite3


from adapt.config import PIPELINE_CONFIG, get_grid_kwargs, get_output_path
from adapt.radar.loader import RadarDataLoader
from adapt.radar.cell_segmenter import RadarCellSegmenter
from adapt.radar.cell_analyzer import RadarCellAnalyzer
from adapt.radar.cell_projector import RadarCellProjector
from adapt.radar.radar_utils import compute_all_cell_centroids, compute_time_interval, get_grid_spacing

logger = logging.getLogger(__name__)


class RadarProcessor(threading.Thread):
    """#> Processes radar data: segment → analyze → project → output.

    #> This class runs in Thread B and performs the full scientific pipeline:
    1. Read the netcdf gridded file if already processed and availbale, else read Level-II files and regrid. all this is done via RadarDataLoader, the grid_ds is then provided for futher processing.
    2. Segment cells via RadarCellSegmenter, - extract the z_leve for entire file, keep all the attributes in the ds and then call segmenter. segmenter gives the same Ddatset back with added fields.
    3. Regsiter and Project cells forward via RadarCellProjector, - segmentation ds will be input to this stage and output will be the same DS with added fields, like flow, and added projected cells with required number of ofsset dims. Additional calculation we do to generate the registration labels for the cells in image 1 to image2,
    that mean, we now keep these registration and projections in same array, showing projection (actually registration), that is from image 1, image2 at 0th index and other projections from that step onwards, and keep segmentation cell_labels as single 2d array showing only the current segmentions. So we will have, in each timestep, the, segmentaion lables array(t0, x, y), (results of segmentation for the given t0), and  projected lables array(offset, x,y), offset for t1, t1 and so on
    projections will be 1-n, and offset for registration lables from t-1
    to t0, will be 0, that is showing projection of objects from image 1 to image2 using the optical flow vectors from image1 to image2. This will also write this into the file to disk and send the signal to plotter queue that the file is ready to read and plot. Note the first file will not have projections or registrations calculation, so those values can be missing. the projection, motion and registration works from the frame 2, other things like segmentation will work for
    the frame 1 and 2 both, so take care that we do not get error for not having two frame for the first frame, the missing data can be stored for the missing variables.
    4. Analyze cell properties via RadarCellAnalyzer, this will take both the DS, grid_ds and analysis_ds and compute the properties and save them in the sqlite database.
    5. Insert results into SQLite database

    SQLite provides efficient storage with ACID transactions and make possible to add
    future tracking UID assignment via random updates.
    Results can be exported to Parquet/duckdn for analysis.
    """

    def __init__(self, input_queue: queue.Queue, config: Dict = None,
                 output_queue: queue.Queue = None,
                 name: str = "RadarProcessor"):
        """Initialize processor.

        Parameters
        ----------
        input_queue : queue.Queue
            Queue of filepaths from downloader.
        config : dict, optional
            Pipeline configuration. If None, uses PIPELINE_CONFIG.
        output_queue : queue.Queue, optional
            Queue to push segmentation results for visualization.
        name : str
            Thread name.
        """
        super().__init__(daemon=True, name=name)

        self.input_queue = input_queue
        self.config = config or PIPELINE_CONFIG
        self.output_queue = output_queue  # For plotter thread
        self._stop_event = threading.Event()

        # Build sub-module configs with global section included
        global_cfg = self.config.get("global", {})
        
        segmenter_cfg = {**self.config.get("segmenter", {}), "global": global_cfg}
        analyzer_cfg = {**self.config.get("analyzer", {}), "global": global_cfg}
        projector_cfg = {**self.config.get("projector", {}), "global": global_cfg}

        # Initialize processing modules
        self.loader = RadarDataLoader(self.config)
        self.segmenter = RadarCellSegmenter(segmenter_cfg)
        self.analyzer = RadarCellAnalyzer(analyzer_cfg)
        self.projector = RadarCellProjector(projector_cfg)

        # Keep last 2 datasets so we can compute flow and projections
        self.dataset_history = []  # List of (filepath, ds) tuples
        self.max_history = 2

        # SQLite database connection
        self.db_path = self._get_db_path()
        self.db_conn = None
        self.output_lock = threading.Lock()
        self._init_database()

    def _get_db_path(self) -> Path:
        """Get SQLite database path at radar-level (persists across runs/dates).

        Database is stored at: analysis/{radar_id}_cells_statistics.db
        This ensures all cells from a pipeline run (even across multiple dates)
        go into the same database.
        """
        output_dirs = self.config.get("output_dirs", {})
        radar_id = self.config.get("downloader", {}).get("radar_id", "UNKNOWN")

        if output_dirs:
            # Store at radar level, not date level
            # This ensures persistence across runs and date boundaries
            analysis_dir = Path(output_dirs.get("analysis", "."))
            analysis_dir.mkdir(parents=True, exist_ok=True)
            db_filename = f"{radar_id}_cells_statistics.db"
            return analysis_dir / db_filename
        else:
            # Fallback for old configs
            db_dir = Path(self.config.get("working_dir", ".")) / "analysis"
            db_dir.mkdir(parents=True, exist_ok=True)
            return db_dir / f"{radar_id}_cells.db"

    def _init_database(self):
        """Initialize SQLite database (schema created on first insert)."""
        self.db_conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.db_initialized = False
        logger.info(f"📊 SQLite database initialized: {self.db_path}")

    def _create_indexes(self):
        """Create indexes after table creation."""
        try:
            self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON cells(timestamp)")
            self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_cell_id ON cells(cell_id)")
            self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_filepath ON cells(filepath)")
            self.db_conn.commit()
            logger.debug("Created database indexes")
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")

    def stop(self):
        """Signal processor to stop gracefully."""
        self._stop_event.set()

    def close_database(self):
        """Close database connection (call after final save)."""
        if self.db_conn:
            self.db_conn.close()
            logger.info("Database connection closed")

    def stopped(self):
        """Check if processor should stop."""
        return self._stop_event.is_set()

    def process_file(self, filepath: str) -> bool:
        """Process a single radar file through the full pipeline.

        Parameters
        ----------
        filepath : str
            Path to Level-II radar file.

        Returns
        -------
        success : bool
            True if processing succeeded, False otherwise.
        """
        try:
            # Extract path from dict if needed
            if isinstance(filepath, dict):
                filepath = filepath["path"]

            # Check if file is already fully processed (skip re-processing)
            file_id = Path(filepath).stem
            tracker = self.config.get("file_tracker")
            if tracker and tracker.should_process(file_id, "analyzed") is False:
                # File already analyzed - check if plotting is also done
                if tracker.should_process(file_id, "plotted") is False:
                    logger.info("⏭️  Skipping already completed: %s", Path(filepath).name)
                    return True
                else:
                    # Re-queue for plotting only (load existing analysis file)
                    logger.info("📊 Already analyzed, re-queuing for plot: %s", Path(filepath).name)
                    # Find the analysis file and queue it for plotting
                    output_dirs = self.config.get("output_dirs", {})
                    radar_id = self.config.get("downloader", {}).get("radar_id", "UNKNOWN")
                    if output_dirs and self.output_queue:
                        from adapt.setup_directories import get_analysis_path
                        from datetime import datetime
                        try:
                            parts = file_id.split('_')
                            datetime_str = parts[0][-8:] + parts[1]
                            scan_time = datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
                        except:
                            scan_time = datetime.now()

                        seg_nc_path = get_analysis_path(
                            output_dirs=output_dirs,
                            radar_id=radar_id,
                            analysis_type="segmentation",
                            timestamp=scan_time,
                            filename=f"{file_id}_analysis.nc"
                        )
                        if seg_nc_path.exists():
                            try:
                                item = {
                                    'segmentation_nc': seg_nc_path,
                                    'gridnc_file': None,
                                    'radar_id': radar_id,
                                    'timestamp': scan_time,
                                }
                                self.output_queue.put_nowait(item)
                                logger.debug("Re-queued for plotting: %s", seg_nc_path.name)
                            except queue.Full:
                                pass
                    return True

            logger.info("Processing: %s", Path(filepath).name)

            # ================================================================
            # STEP 1: Load and Regrid (or load existing NetCDF)
            # ================================================================
            grid_kwargs = get_grid_kwargs()
            save_netcdf = self.config.get("regridder", {}).get("save_netcdf", True)

            # Determine NetCDF path
            output_dirs = self.config.get("output_dirs")
            nc_full_path = None
            if output_dirs:
                from adapt.setup_directories import get_netcdf_path
                from datetime import datetime

                radar_id = self.config.get("downloader", {}).get("radar_id", "UNKNOWN")
                nc_filename = Path(filepath).stem

                # Extract scan_time from filename (e.g., KDIX20251127_040228_V06)
                try:
                    parts = nc_filename.split('_')
                    datetime_str = parts[0][-8:] + parts[1]  # YYYYMMDD + HHMMSS
                    scan_time = datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
                except:
                    scan_time = datetime.now()

                nc_full_path = get_netcdf_path(output_dirs, radar_id, nc_filename, scan_time=scan_time)
                output_dir = str(nc_full_path.parent)
            else:
                output_dir = self.config.get("downloader", {}).get("output_dir")

            # Check if regridded NetCDF already exists
            ds = None
            if nc_full_path and nc_full_path.exists() and nc_full_path.stat().st_size > 1000:
                # NetCDF exists → load it via PyART to preserve Grid structure
                try:
                    import pyart
                    grid = pyart.io.read_grid(str(nc_full_path))
                    ds = grid.to_xarray()
                    logger.debug("Loaded existing NetCDF: %s", nc_full_path.name)
                except Exception as e:
                    logger.warning("Failed to load existing NetCDF %s: %s", nc_full_path.name, e)
                    ds = None  # Fall through to regridding

            # If NetCDF doesn't exist or failed to load → regrid from NEXRAD
            if ds is None:
                ds = self.loader.load_and_regrid(filepath, grid_kwargs=grid_kwargs,
                                                save_netcdf=save_netcdf, output_dir=output_dir)

            if ds is None:
                logger.warning("Failed to load/regrid: %s", filepath)
                return False

            logger.debug("Loaded and regridded: %s", Path(filepath).name)

            # ================================================================
            # STEP 1.5: Extract 2D slice at z-level
            # ================================================================
            # From this point forward, we work with 2D data only
            ds_2d = self._extract_2d_slice(ds)
            logger.debug(f"Extracted 2D slice at z-level, shape: {ds_2d.dims}")

            # ================================================================
            # STEP 2: Segment Cells (2D)
            # ================================================================
            ds_2d = self.segmenter.segment(ds_2d)

            var_names = self.config.get("global", {}).get("var_names", {})
            labels_name = var_names.get("cell_labels", "cell_labels")

            if labels_name not in ds_2d.data_vars:
                logger.warning("Segmentation failed for: %s", filepath)
                return False

            num_cells = int(ds_2d[labels_name].max().item())
            logger.info("Segmented: %d cells", num_cells)

            # Save segmentation NetCDF for visualization
            seg_nc_path = self._save_segmentation_netcdf(ds_2d, filepath)

            # Push to plotter queue if available (use scan_time, not current time)
            if seg_nc_path and self.output_queue:
                try:
                    item = {
                        'segmentation_nc': seg_nc_path,
                        'gridnc_file': None,  # Could add regridded gridnc path if desired
                        'radar_id': self.config.get("downloader", {}).get("radar_id", "UNKNOWN"),
                        'timestamp': scan_time,  # Use scan time for proper date organization
                    }
                    self.output_queue.put_nowait(item)
                    logger.debug(f"Pushed segmentation to plotter queue: {seg_nc_path}")
                except queue.Full:
                    logger.debug("Plotter queue full, skipping frame")

            # ================================================================
            # STEP 3: Analyze Cells (2D)
            # ================================================================
            z_level = self.config.get("global", {}).get("z_level", 2000)
            df_cells = self.analyzer.extract(ds_2d, z_level=z_level)

            # Update tracker for regridded stage (do this before early return)
            tracker = self.config.get("file_tracker")
            file_id = Path(filepath).stem
            if tracker and nc_full_path:
                tracker.mark_stage_complete(file_id, "regridded", path=nc_full_path)

            # Skip if no cells detected (but still mark as analyzed with 0 cells)
            if df_cells.empty:
                logger.debug("No cells detected in: %s", Path(filepath).name)
                if tracker:
                    tracker.mark_stage_complete(file_id, "analyzed", num_cells=0)
                return True

            # Add metadata
            df_cells["filepath"] = str(filepath)  # Convert Path to string for Parquet compatibility
            df_cells["timestamp"] = pd.Timestamp.now()

            # Add centroids
            var_names = self.config.get("global", {}).get("var_names", {})
            labels_name = var_names.get("cell_labels", "cell_labels")
            labels = ds_2d[labels_name].values
            centroids = compute_all_cell_centroids(labels)
            df_cells["centroid_y"] = df_cells["cell_id"].map(lambda cid: centroids.get(cid, (np.nan, np.nan))[0])
            df_cells["centroid_x"] = df_cells["cell_id"].map(lambda cid: centroids.get(cid, (np.nan, np.nan))[1])

            logger.debug("Analyzed: %d cells with properties", len(df_cells))

            # Log detailed cell statistics
            self._log_cell_statistics(df_cells)

            # ================================================================
            # STEP 4: Insert into SQLite database
            # ================================================================
            with self.output_lock:
                # On first insert, create table schema from DataFrame
                if not self.db_initialized:
                    df_cells.to_sql('cells', self.db_conn, if_exists='replace', index=False)
                    self._create_indexes()
                    self.db_initialized = True
                    logger.info("📊 Database schema created from first DataFrame")
                else:
                    df_cells.to_sql('cells', self.db_conn, if_exists='append', index=False)

            logger.info("✓ Successfully processed: %s (saved %d cells to DB)", Path(filepath).name, len(df_cells))

            # Update tracker for analyzed stage (regridded was updated earlier)
            if tracker:
                tracker.mark_stage_complete(file_id, "analyzed", num_cells=len(df_cells))

            return True

        except Exception as e:
            logger.exception("Error processing %s", filepath)

            # Mark as failed in tracker
            tracker = self.config.get("file_tracker")
            if tracker:
                file_id = Path(filepath).stem
                tracker.mark_stage_complete(file_id, "analyzed", error=str(e))

            return False

    def run(self):
        """Main processor loop: read queue, process files."""
        logger.info("Processor started, waiting for files...")

        while not self.stopped():
            try:
                # Get next file from queue (block for 1 second)
                try:
                    filepath = self.input_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # Process file
                success = self.process_file(filepath)

                # Mark task as done
                self.input_queue.task_done()

            except Exception as e:
                logger.exception("Processor error")

        logger.info("Processor stopped")

    # ...existing code...

    def get_results(self) -> pd.DataFrame:
        """Get accumulated results as DataFrame (thread-safe).

        Returns
        -------
        df : pd.DataFrame
            DataFrame with all cells from database.
        """
        with self.output_lock:
            # Don't query before schema is created
            if not self.db_initialized:
                return pd.DataFrame()

            try:
                return pd.read_sql("SELECT * FROM cells", self.db_conn)
            except Exception as e:
                logger.warning("Failed to read from database: %s", e)
                return pd.DataFrame()

    def save_results(self, filepath: str = None):
        """Export results to Parquet.

        Parameters
        ----------
        filepath : str, optional
            Output filepath. If None, uses config-derived path.
        """
        if filepath is None:
            filepath = get_output_path()

        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with self.output_lock:
                cursor = self.db_conn.execute("SELECT COUNT(*) FROM cells")
                count = cursor.fetchone()[0]

                if count > 0:
                    df = pd.read_sql("SELECT * FROM cells", self.db_conn)
                    df.to_parquet(filepath, engine='pyarrow',
                                  compression=self.config["output"]["compression"],
                                  index=False)
                    logger.info("Exported %d rows to: %s", count, filepath)
                else:
                    logger.warning("No results to export")

        except Exception as e:
            logger.exception("Failed to export results")

    def _log_cell_statistics(self, df_cells: pd.DataFrame):
        """Log detailed cell statistics for each detected cell.

        Parameters
        ----------
        df_cells : pd.DataFrame
            DataFrame with cell properties from analyzer.
        """
        try:
            if df_cells.empty:
                return

            # Extract key statistics from DataFrame
            num_cells = len(df_cells)

            # Build statistics log line
            stats_parts = [f"Cells: {num_cells}"]

            # Area statistics
            if "area_km2" in df_cells.columns:
                area_vals = df_cells["area_km2"].dropna()
                if len(area_vals) > 0:
                    stats_parts.append(
                        f"Area [km²] - min={area_vals.min():.1f}, "
                        f"max={area_vals.max():.1f}, "
                        f"mean={area_vals.mean():.1f}"
                    )

            # Reflectivity statistics
            if "reflectivity_mean" in df_cells.columns:
                refl_mean_vals = df_cells["reflectivity_mean"].dropna()
                if len(refl_mean_vals) > 0:
                    stats_parts.append(
                        f"Refl-mean [dBZ] - min={refl_mean_vals.min():.1f}, "
                        f"max={refl_mean_vals.max():.1f}, "
                        f"mean={refl_mean_vals.mean():.1f}"
                    )

            if "reflectivity_max" in df_cells.columns:
                refl_max_vals = df_cells["reflectivity_max"].dropna()
                if len(refl_max_vals) > 0:
                    stats_parts.append(
                        f"Refl-max [dBZ] - min={refl_max_vals.min():.1f}, "
                        f"max={refl_max_vals.max():.1f}, "
                        f"mean={refl_max_vals.mean():.1f}"
                    )

            # Log summary
            summary_msg = " | ".join(stats_parts)
            logger.info("📊 Cell Statistics: %s", summary_msg)

            # Log individual cell details for small number of cells
            if num_cells <= 5:
                for idx, row in df_cells.iterrows():
                    cell_id = row.get("cell_id", "?")
                    area = row.get("area_km2", np.nan)
                    refl_mean = row.get("reflectivity_mean", np.nan)
                    refl_max = row.get("reflectivity_max", np.nan)

                    detail_msg = f"Cell #{cell_id}: area={area:.1f} km², refl_mean={refl_mean:.1f} dBZ, refl_max={refl_max:.1f} dBZ"
                    logger.info("  └─ %s", detail_msg)

        except Exception as e:
            logger.debug("Could not log cell statistics: %s", e)

    def _extract_2d_slice(self, ds: xr.Dataset) -> xr.Dataset:
        """Extract 2D slice at configured z-level.
        
        Parameters
        ----------
        ds : xr.Dataset
            Full 3D dataset from loader
            
        Returns
        -------
        xr.Dataset
            2D dataset with all variables sliced at z-level
        """
        global_cfg = self.config.get("global", {})
        z_level = global_cfg.get("z_level", 2000)
        coord_names = global_cfg.get("coord_names", {})
        time_name = coord_names.get("time", "time")
        z_name = coord_names.get("z", "z")
        
        # Find z-level index
        z_idx = int(np.argmin(np.abs(ds[z_name].values - z_level)))
        
        # Extract 2D slice for all 3D variables
        ds_2d = xr.Dataset()
        for var_name in ds.data_vars:
            var = ds[var_name]
            # Check if variable has time and z dimensions
            if time_name in var.dims and z_name in var.dims:
                # Slice to 2D
                ds_2d[var_name] = var.isel({time_name: 0, z_name: z_idx})
            else:
                # Keep as-is (already 2D or scalar)
                ds_2d[var_name] = var
        
        # Copy coordinates (only y, x for 2D)
        ds_2d = ds_2d.assign_coords({
            "y": ds.y,
            "x": ds.x,
        })
        
        # Copy attributes
        ds_2d.attrs.update(ds.attrs)
        ds_2d.attrs["z_level_m"] = float(ds[z_name].values[z_idx])
        
        logger.debug(f"Extracted 2D slice at z={ds[z_name].values[z_idx]:.0f}m (index {z_idx})")
        return ds_2d

    def _save_segmentation_netcdf(self, ds: xr.Dataset, filepath: str):
        """Save analysis results to NetCDF for visualization.

        Parameters
        ----------
        ds : xr.Dataset
            2D dataset with cell_labels and reflectivity at z-level
        filepath : str
            Original radar file path (used to derive output name)
        """
        try:
            output_dirs = self.config.get("output_dirs")
            if not output_dirs:
                return

            from datetime import datetime

            # Extract scan_time from filepath for YYYYMMDD directory
            try:
                filename = Path(filepath).stem
                parts = filename.split('_')
                datetime_str = parts[0][-8:] + parts[1]  # YYYYMMDD + HHMMSS
                scan_time = datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
            except:
                scan_time = datetime.now()

            from adapt.setup_directories import get_analysis_path
            radar_id = self.config.get("downloader", {}).get("radar_id", "UNKNOWN")

            # Create output path
            filename_stem = Path(filepath).stem
            seg_nc_filename = f"{filename_stem}_analysis.nc"
            seg_nc_path = get_analysis_path(
                output_dirs,
                radar_id=radar_id,
                analysis_type="netcdf",
                timestamp=scan_time,
                filename=seg_nc_filename
            )
            seg_nc_path.parent.mkdir(parents=True, exist_ok=True)

            # Update rolling history with current 2D dataset
            self.dataset_history.append((filepath, ds))
            if len(self.dataset_history) > self.max_history:
                self.dataset_history.pop(0)

            # Get dataset to save (with projections if available)
            ds_to_save = ds
            
            # Compute optical flow projections if we have 2+ frames
            # Projector expects 2D datasets
            if len(self.dataset_history) >= 2:
                try:
                    ds_prev_path, ds_prev = self.dataset_history[-2]
                    ds_curr_path, ds_curr = self.dataset_history[-1]

                    # Projector returns 2D ds with cell_projections, flow_u, flow_v added
                    ds_with_proj = self.projector.project([ds_prev, ds_curr])

                    if ds_with_proj is not None:
                        ds_to_save = ds_with_proj
                        logger.info(f"✓ Projections computed: {list(ds_with_proj.data_vars)}")
                except Exception as e:
                    logger.error(f"Optical flow projection failed: {e}", exc_info=True)
            else:
                logger.debug("First frame: optical flow not available (need 2+ datasets)")

            # Add metadata
            ds_to_save.attrs.update({
                "source": str(filepath),
                "radar_id": radar_id,
                "description": "Radar analysis with segmentation and projections"
            })

            # Save to NetCDF
            seg_nc_path.parent.mkdir(parents=True, exist_ok=True)
            ds_to_save.to_netcdf(
                seg_nc_path,
                mode='w',
                engine='netcdf4',
                format='NETCDF4'
            )
            ds_to_save.close()

            # Log what was saved
            components = list(ds_to_save.data_vars.keys())
            logger.info(f"✓ Analysis saved: {seg_nc_path.name} [{', '.join(components)}]")

            return str(seg_nc_path)

        except Exception as e:
            logger.warning(f"Could not save segmentation NetCDF: {e}")
            return None


class MultiFileProcessor:
    """Process multiple radar files sequentially (non-threaded).

    Useful for batch processing historical data or testing.
    """

    def __init__(self, config: Dict = None):
        """Initialize processor.

        Parameters
        ----------
        config : dict, optional
            Pipeline configuration.
        """
        self.config = config or PIPELINE_CONFIG

        # Build sub-module configs with global section included
        global_cfg = self.config.get("global", {})
        segmenter_cfg = {**self.config.get("segmenter", {}), "global": global_cfg}
        analyzer_cfg = {**self.config.get("analyzer", {}), "global": global_cfg}

        # Initialize processing modules
        self.loader = RadarDataLoader(self.config)
        self.segmenter = RadarCellSegmenter(segmenter_cfg)
        self.analyzer = RadarCellAnalyzer(analyzer_cfg)

        # SQLite database connection
        self.db_path = self._get_db_path()
        self.db_conn = sqlite3.connect(str(self.db_path))
        self.db_initialized = False
        self._init_database()

    def _get_db_path(self) -> Path:
        """Get SQLite database path using YYYYMMDD/RADAR_ID structure."""
        output_dirs = self.config.get("output_dirs", {})
        radar_id = self.config.get("downloader", {}).get("radar_id", "UNKNOWN")

        if output_dirs:
            # Use get_analysis_path for consistent structure
            from adapt.setup_directories import get_analysis_path
            from datetime import datetime

            db_filename = f"{radar_id}_cells_statistics.db"
            db_path = get_analysis_path(
                output_dirs=output_dirs,
                radar_id=radar_id,
                analysis_type="db",
                timestamp=datetime.now(),
                filename=db_filename
            )
            return db_path
        else:
            # Fallback for old configs
            db_dir = Path(self.config.get("working_dir", ".")) / "analysis"
            db_dir.mkdir(parents=True, exist_ok=True)
            return db_dir / f"{radar_id}_cells.db"

    def _init_database(self):
        """Initialize SQLite database (schema created on first insert)."""
        logger.info(f"📊 SQLite database initialized: {self.db_path}")

    def process_files(self, filepaths: List[str]) -> pd.DataFrame:
        """Process multiple files and accumulate results.

        Parameters
        ----------
        filepaths : list of str
            Paths to radar files.

        Returns
        -------
        df : pd.DataFrame
            Accumulated results.
        """
        for filepath in filepaths:
            try:
                logger.info("Processing: %s", Path(filepath).name)

                # Load and regrid
                grid_kwargs = get_grid_kwargs()
                ds = self.loader.load_and_regrid(filepath, grid_kwargs=grid_kwargs)

                if ds is None:
                    logger.warning("Failed to load: %s", filepath)
                    continue

                # Extract 2D slice at z-level
                ds_2d = self._extract_2d_slice(ds)

                # Segment (2D)
                ds_2d = self.segmenter.segment(ds_2d)

                var_names = self.config.get("global", {}).get("var_names", {})
                labels_name = var_names.get("cell_labels", "cell_labels")

                if labels_name not in ds_2d.data_vars:
                    logger.warning("Segmentation failed: %s", filepath)
                    continue

                # Analyze (2D)
                z_level = self.config.get("global", {}).get("z_level", 2000)
                df_cells = self.analyzer.extract(ds_2d, z_level=z_level)

                # Handle case when no cells detected
                if df_cells.empty:
                    logger.info("⚠️ No cells detected in %s, skipping", Path(filepath).name)
                    continue

                df_cells["filepath"] = filepath
                df_cells["timestamp"] = pd.Timestamp.now()

                # Add centroids
                labels = ds_2d[labels_name].values
                centroids = compute_all_cell_centroids(labels)
                df_cells["centroid_y"] = df_cells["cell_id"].map(
                    lambda cid: centroids.get(cid, (np.nan, np.nan))[0])
                df_cells["centroid_x"] = df_cells["cell_id"].map(
                    lambda cid: centroids.get(cid, (np.nan, np.nan))[1])

                # Save segmentation NetCDF for visualization
                seg_nc_path = self._save_segmentation_netcdf(ds_2d, filepath)

                # Generate plot if visualization enabled
                if self.config.get("visualization", {}).get("enabled", False) and seg_nc_path:
                    self._generate_plot(seg_nc_path, filepath)

                # Insert into database
                if not self.db_initialized:
                    df_cells.to_sql('cells', self.db_conn, if_exists='replace', index=False)
                    self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON cells(timestamp)")
                    self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_cell_id ON cells(cell_id)")
                    self.db_conn.commit()
                    self.db_initialized = True
                    logger.info("📊 Database schema created from first DataFrame")
                else:
                    df_cells.to_sql('cells', self.db_conn, if_exists='append', index=False)
                logger.info("✓ Processed: %s (%d cells)", Path(filepath).name, len(df_cells))

            except Exception as e:
                logger.exception("Error processing %s", filepath)
                continue

        # Return all results from database
        return pd.read_sql("SELECT * FROM cells", self.db_conn)

    def save_results(self, filepath: str = None):
        """Export results to Parquet.

        Parameters
        ----------
        filepath : str, optional
            Output filepath. If None, uses config-derived path.
        """
        if filepath is None:
            filepath = get_output_path()

        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            cursor = self.db_conn.execute("SELECT COUNT(*) FROM cells")
            count = cursor.fetchone()[0]

            if count > 0:
                df = pd.read_sql("SELECT * FROM cells", self.db_conn)
                df.to_parquet(filepath, engine='pyarrow',
                              compression=self.config["output"]["compression"],
                              index=False)
                logger.info("Exported %d rows to: %s", count, filepath)
            else:
                logger.warning("No results to export")

        except Exception as e:
            logger.exception("Failed to export results")

    def _extract_2d_slice(self, ds: xr.Dataset) -> xr.Dataset:
        """Extract 2D slice at configured z-level (shared with RadarProcessor)."""
        global_cfg = self.config.get("global", {})
        z_level = global_cfg.get("z_level", 2000)
        coord_names = global_cfg.get("coord_names", {})
        time_name = coord_names.get("time", "time")
        z_name = coord_names.get("z", "z")
        
        z_idx = int(np.argmin(np.abs(ds[z_name].values - z_level)))
        
        ds_2d = xr.Dataset()
        for var_name in ds.data_vars:
            var = ds[var_name]
            if time_name in var.dims and z_name in var.dims:
                ds_2d[var_name] = var.isel({time_name: 0, z_name: z_idx})
            else:
                ds_2d[var_name] = var
        
        ds_2d = ds_2d.assign_coords({"y": ds.y, "x": ds.x})
        ds_2d.attrs.update(ds.attrs)
        ds_2d.attrs["z_level_m"] = float(ds[z_name].values[z_idx])
        
        return ds_2d

    def _save_segmentation_netcdf(self, ds: xr.Dataset, filepath: str):
            """Save analysis results to NetCDF for visualization.

            Parameters
            ----------
            ds : xr.Dataset
                Dataset with cell_labels and reflectivity (already at z-level)
            filepath : str
                Original radar file path (used to derive output name)
            """
            try:
                output_dirs = self.config.get("output_dirs")
                if not output_dirs:
                    return

                from adapt.setup_directories import get_analysis_path
                radar_id = self.config.get("downloader", {}).get("radar_id", "UNKNOWN")

                # Create output path
                filename_stem = Path(filepath).stem
                seg_nc_path = get_analysis_path(
                    output_dirs,
                    radar_id=radar_id,
                    analysis_type="netcdf",
                    timestamp=pd.Timestamp.now().to_pydatetime(),
                )
                seg_nc_path = seg_nc_path.with_name(f"{filename_stem}_analysis.nc")
                seg_nc_path.parent.mkdir(parents=True, exist_ok=True)

                # Add metadata
                ds.attrs.update({
                    "source": str(filepath),
                    "radar_id": radar_id,
                })

                # Save dataset as-is (projector already added cell_projections if available)
                ds.to_netcdf(seg_nc_path, engine='netcdf4', unlimited_dims=['time'])

                logger.info(f"✓ Analysis saved: {seg_nc_path} [{list(ds.data_vars.keys())}]")

                return str(seg_nc_path)

            except Exception as e:
                logger.warning(f"Could not save segmentation NetCDF: {e}")
                return None

    def _generate_plot(self, seg_nc_path: str, filepath: str):
        """Generate plot from segmentation NetCDF.

        Parameters
        ----------
        seg_nc_path : str
            Path to segmentation NetCDF file
        filepath : str
            Original radar file path (for metadata)
        """
        try:
            from adapt.visualization.plotter import RadarPlotter

            plotter = RadarPlotter(config=self.config)
            plot_path = plotter.plot_from_netcdf(seg_nc_path)

            if plot_path:
                logger.info(f"📊 Plot saved: {plot_path}")

        except Exception as e:
            logger.warning(f"Could not generate plot: {e}")


if __name__ == "__main__":
    # Logging configured by pipeline orchestrator
    pass  # Placeholder for demo code below
    print("Processor classes loaded. Use in pipeline orchestration.")
