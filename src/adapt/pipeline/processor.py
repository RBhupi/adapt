"""Radar data processing pipeline.

Processes radar files through segmentation, projection, and analysis stages.
Reads Level-II data or gridded NetCDF, segments cells, projects motion,
computes statistics, and persists results to NetCDF and SQLite.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, TYPE_CHECKING
import threading
import queue
from datetime import datetime


import pandas as pd
import numpy as np
import xarray as xr
import sqlite3


from adapt.radar.loader import RadarDataLoader
from adapt.radar.cell_segmenter import RadarCellSegmenter
from adapt.radar.cell_analyzer import RadarCellAnalyzer
from adapt.radar.cell_projector import RadarCellProjector
from adapt.radar.radar_utils import compute_all_cell_centroids
from adapt.contracts import (
    ContractViolation,
    assert_gridded,
    assert_segmented,
    assert_projected,
    assert_analysis_output,
)

if TYPE_CHECKING:
    from adapt.schemas import InternalConfig

__all__ = ['RadarProcessor']

logger = logging.getLogger(__name__)


class RadarProcessor(threading.Thread):
    """Processes radar files through the complete scientific analysis pipeline.

    This worker thread runs in the background, receiving filepaths from the
    downloader queue and performing all scientific analysis: loading, regridding,
    segmentation, motion projection, and cell statistics extraction.

    **Processing Pipeline:**

    For each file, the processor performs (in order):

    1. **Load & Regrid**: Reads NEXRAD Level-II data or cached gridded NetCDF,
       converts to Cartesian grid using Cressman weighting and ARM Py-ART.

    2. **Segmentation**: Identifies storm cells using reflectivity threshold
       and morphological filtering. Outputs cell_labels array.

    3. **Motion Projection** (frame 2+): Computes optical flow between current
       and previous frame, projects cells forward by 1-5 timesteps. First frame
       has no projections (only segmentation).

    4. **Cell Analysis**: Extracts properties from each cell:
       - Geometry: area, centroid (mass-weighted and geometric)
       - Reflectivity: max, mean, 95th percentile
       - Dimensions: length, width, aspect ratio
       Stores in SQLite database for later analysis.

    5. **Persistence**: Writes segmentation + projections to NetCDF
       (analysis/{radar_id}_analysis_*_segmentation.nc) and queues file
       for visualization.

    **Output Files:**

    - **Gridded NetCDF**: gridded/{radar_id}_*_gridded.nc (regridded data)
    - **Analysis NetCDF**: analysis/{radar_id}_*_analysis_segmentation.nc
      (segmentation + projections + flow fields)
    - **SQLite Database**: analysis/{radar_id}_cells_statistics.db
      (cell properties for all files)

    **Database Schema:**

    SQLite table `cells` (one row per detected cell):
    - Temporal: time, time_volume_start, time_volume_end
    - Identification: cell_label, radar_id, file_id
    - Geometry: cell_area_sqkm, centroids (lat/lon and x/y)
    - Reflectivity: max_dbz, mean_dbz, percentile_95
    - Dimensions: length_km, width_km, aspect_ratio
    - Motion: heading_x, heading_y, projection_distances (1-5 steps)

    **Frame History:**

    Maintains a 2-frame rolling window to compute optical flow. This enables
    motion estimation between frame N and frame N+1. Frame 1 has only
    segmentation; projections begin at frame 2.

    **Thread Safety:**

    Uses SQLite WAL mode for concurrent read access. Main thread can query
    results while processor is still running.

    Example usage (typically called by orchestrator)::

        processor = RadarProcessor(
            input_queue=downloader_queue,
            config=config,
            output_queue=plotter_queue
        )
        processor.start()
        ...
        processor.stop()
        df = processor.get_results()
    """

    def __init__(self, input_queue: queue.Queue, config: "InternalConfig",
                 output_dirs: Dict[str, Path],
                 output_queue: queue.Queue = None,
                 file_tracker = None,
                 name: str = "RadarProcessor"):
        """Initialize processor with validated configuration.

        Parameters
        ----------
        input_queue : queue.Queue
            Queue of filepaths from downloader thread. Processor pops filepaths
            and processes them. None signals shutdown.

        config : InternalConfig
            Fully validated runtime configuration.
            
        output_dirs : dict
            Output directory paths (from setup_output_directories):
            - nexrad: NEXRAD Level-II files
            - gridded: Regridded NetCDF files
            - analysis: Analysis NetCDF files
            - plots: Visualization PNG files
            - logs: Log files

        output_queue : queue.Queue, optional
            Queue to send analysis files to plotter thread. One dict per file:
            
            - `segmentation_nc`: Path to analysis NetCDF file
            - `radar_id`: Radar identifier
            - `timestamp`: Datetime of the scan
            
            If None, no plotting occurs (analysis-only mode).
            
        file_tracker : FileProcessingTracker, optional
            Optional file processing tracker to skip already completed files.

        name : str, optional
            Thread name for logging (default: "RadarProcessor").

        Raises
        ------
        Exception
            If database initialization fails or output directories don't exist.
        """
        super().__init__(daemon=True, name=name)

        self.input_queue = input_queue
        self.config = config
        self.output_dirs = output_dirs
        self.output_queue = output_queue  # For plotter thread
        self.file_tracker = file_tracker
        self._stop_event = threading.Event()

        # Initialize processing modules
        self.loader = RadarDataLoader(self.config)
        self.segmenter = RadarCellSegmenter(self.config)
        self.analyzer = RadarCellAnalyzer(self.config)
        self.projector = RadarCellProjector(self.config)

        # Keep last N datasets so we can compute flow and projections
        # Uses max_history from config instead of hardcoding
        self.dataset_history = []  # List of (filepath, ds) tuples
        self.max_history = self.config.processor.max_history

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
        
        radar_id and output_dir are required fields from InternalConfig.downloader.
        """
        radar_id = self.config.downloader.radar_id  # Required field, no fallback
        analysis_dir = Path(self.output_dirs["analysis"])
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Use filename pattern from config
        db_filename = self.config.processor.db_filename_pattern.format(radar_id=radar_id)
        return analysis_dir / db_filename

    def _init_database(self):
        """Initialize SQLite database (schema created on first insert)."""
        self.db_conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.db_initialized = False
        logger.info(f"Database initialized: {self.db_path}")

    def _create_cells_table(self, df_cells: pd.DataFrame):
        """Create cells table with explicit schema including all expected columns.
        
        Creates columns for all possible centroid types (geom, mass, maxdbz, registration, projections),
        heading statistics, and radar variables. Missing columns are filled with NULL.
        
        Primary Key: (time, cell_label) where time = time_volume_start
        Includes time_volume_start, time, and time_volume_end for temporal tracking.
        
        This ensures consistent schema across all files regardless of which features are available.
        """
        # Define all expected columns with their types
        # First scan may not have projections (needs 2+ frames), but schema includes them for consistency
        expected_columns = {
            # Temporal keys - all TIMESTAMP
            "time_volume_start": "TIMESTAMP NOT NULL",  # Start of radar volume scan
            "time": "TIMESTAMP NOT NULL",  # Analysis time (= time_volume_start)
            "time_volume_end": "TIMESTAMP",  # End of radar volume scan (future: when available)
            # Cell identifier
            "cell_label": "INTEGER NOT NULL",
            # Basic properties
            "cell_area_sqkm": "REAL",
            # Geometric centroid (center of mass)
            "cell_centroid_geom_x": "INTEGER",
            "cell_centroid_geom_y": "INTEGER",
            "cell_centroid_geom_lat": "REAL",
            "cell_centroid_geom_lon": "REAL",
            # Mass-weighted centroid (reflectivity weighted)
            "cell_centroid_mass_x": "INTEGER",
            "cell_centroid_mass_y": "INTEGER",
            "cell_centroid_mass_lat": "REAL",
            "cell_centroid_mass_lon": "REAL",
            # Max reflectivity centroid
            "cell_centroid_maxdbz_x": "INTEGER",
            "cell_centroid_maxdbz_y": "INTEGER",
            "cell_centroid_maxdbz_lat": "REAL",
            "cell_centroid_maxdbz_lon": "REAL",
            # Registration centroid (projection index 0 - frame-to-frame tracking)
            "cell_centroid_registration_x": "INTEGER",
            "cell_centroid_registration_y": "INTEGER",
            "cell_centroid_registration_lat": "REAL",
            "cell_centroid_registration_lon": "REAL",
            # Forward projection centroids (indices 1-5, configurable)
            "cell_centroid_projection1_x": "INTEGER",
            "cell_centroid_projection1_y": "INTEGER",
            "cell_centroid_projection1_lat": "REAL",
            "cell_centroid_projection1_lon": "REAL",
            "cell_centroid_projection2_x": "INTEGER",
            "cell_centroid_projection2_y": "INTEGER",
            "cell_centroid_projection2_lat": "REAL",
            "cell_centroid_projection2_lon": "REAL",
            "cell_centroid_projection3_x": "INTEGER",
            "cell_centroid_projection3_y": "INTEGER",
            "cell_centroid_projection3_lat": "REAL",
            "cell_centroid_projection3_lon": "REAL",
            "cell_centroid_projection4_x": "INTEGER",
            "cell_centroid_projection4_y": "INTEGER",
            "cell_centroid_projection4_lat": "REAL",
            "cell_centroid_projection4_lon": "REAL",
            "cell_centroid_projection5_x": "INTEGER",
            "cell_centroid_projection5_y": "INTEGER",
            "cell_centroid_projection5_lat": "REAL",
            "cell_centroid_projection5_lon": "REAL",
            # Motion vector statistics
            "cell_heading_x_mean": "REAL",
            "cell_heading_y_mean": "REAL",
            # Projection centroids as JSON (compact storage)
            "cell_projection_centroids_json": "TEXT",
        }
        
        # Add radar variable statistics (from config whitelist)
        radar_vars = self.config.analyzer.radar_variables
        for var in radar_vars:
            for stat in ["mean", "min", "max"]:
                col_name = f"radar_{var}_{stat}"
                expected_columns[col_name] = "REAL"
        
        # Merge with any extra columns from current DataFrame
        # (in case analyzer adds new columns we haven't anticipated)
        col_defs = []
        
        # Add columns in a logical order: keys, geometry, centroids, motion, radar
        column_order = ["time_volume_start", "time", "time_volume_end", "cell_label", "cell_area_sqkm"]
        
        # Add all expected columns
        for col_name in sorted(expected_columns.keys()):
            if col_name not in column_order:
                column_order.append(col_name)
        
        # Add any extra columns from current df that we didn't anticipate
        for col in df_cells.columns:
            if col not in expected_columns:
                column_order.append(col)
        
        # Build column definitions
        for col in column_order:
            if col in expected_columns:
                col_defs.append(f'"{col}" {expected_columns[col]}')
            else:
                # Dynamically determine type for unexpected columns
                if col.endswith("_x") or col.endswith("_y"):
                    col_defs.append(f'"{col}" INTEGER')
                elif col.endswith("_lat") or col.endswith("_lon"):
                    col_defs.append(f'"{col}" REAL')
                elif col.startswith("radar_"):
                    col_defs.append(f'"{col}" REAL')
                elif "_json" in col:
                    col_defs.append(f'"{col}" TEXT')
                else:
                    col_defs.append(f'"{col}" TEXT')
        
        col_defs_str = ",\n    ".join(col_defs)
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS cells (
            {col_defs_str},
            PRIMARY KEY (time, cell_label)
        )
        """
        self.db_conn.execute(create_sql)
        self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_time ON cells(time)")
        self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_time_volume_start ON cells(time_volume_start)")
        self.db_conn.commit()
        logger.info("Database schema created with composite PK (time, cell_label)")
        logger.info(f"   Temporal columns: time_volume_start, time, time_volume_end (TIMESTAMP)")
        logger.info(f"   Total columns: {len(col_defs)} (includes future-proof centroid/heading/projection slots)")

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
        """Process single file: load → regrid → segment → project → analyze → save."""
        try:
            # Step 0: Handle dict input and check tracker for completed/plotting
            if isinstance(filepath, dict):
                filepath = filepath["path"]
            
            # Validate file exists before attempting to process
            file_path = Path(filepath)
            
            file_id = Path(filepath).stem
            tracker = self.file_tracker
            if tracker and tracker.should_process(file_id, "analyzed") is False:
                if tracker.should_process(file_id, "plotted") is False:
                    logger.info("Skipping already completed: %s", Path(filepath).name)
                    return True
                else:
                    self._requeue_for_plotting(file_id, filepath)
                    return True

            logger.info("Processing: %s", Path(filepath).name)

            # Step 1: Load and regrid
            ds, ds_2d, nc_full_path, scan_time = self._load_and_regrid(filepath)
            reflectivity_var = self.config.global_.var_names.reflectivity
            assert_gridded(ds_2d, reflectivity_var)

            # Step 2: Segment
            ds_2d = self.segmenter.segment(ds_2d)
            labels_name = self.config.global_.var_names.cell_labels
            assert_segmented(ds_2d, labels_name)
            num_cells = int(ds_2d[labels_name].max().item())
            logger.info("Segmented: %d cells", num_cells)

            # Step 3: PROJECT (before analyzer so it has projection info)
            ds_2d = self._compute_projections(ds_2d, filepath)
            
            # Contract: if 2+ frames, projections should exist
            if len(self.dataset_history) >= 2:
                assert_projected(ds_2d, self.config.projector.max_projection_steps)

            # Step 4: Save segmentation NetCDF for visualization (no dependency on analysis)
            seg_nc_path = self._save_segmentation_netcdf(ds_2d, filepath, scan_time)
            if seg_nc_path and self.output_queue:
                try:
                    item = {
                        'segmentation_nc': seg_nc_path,
                        'gridnc_file': None,
                        'radar_id': self.config.downloader.radar_id,
                        'timestamp': scan_time,
                    }
                    self.output_queue.put_nowait(item)
                    logger.debug(f"Pushed to plotter queue: {seg_nc_path}")
                except queue.Full:
                    logger.debug("Plotter queue full, skipping frame")

            # Step 5: Analyze (parallel with visualization - no dependency)
            result = self._analyze_and_save(ds_2d, filepath, nc_full_path, num_cells, scan_time)
            return result

        except ContractViolation as e:
            logger.critical("💥 CRITICAL: Pipeline contract violated: %s", e)
            logger.critical("This indicates a bug in pipeline logic. Stopping pipeline.")
            self.stop()  # Signal processor to stop
            tracker = self.file_tracker
            if tracker:
                file_id = Path(filepath).stem
                tracker.mark_stage_complete(file_id, "analyzed", error=f"Contract violation: {e}")
            return False

        except Exception as e:
            logger.exception("Error processing %s", filepath)
            tracker = self.file_tracker
            if tracker:
                file_id = Path(filepath).stem
                tracker.mark_stage_complete(file_id, "analyzed", error=str(e))
            return False

    def _requeue_for_plotting(self, file_id, filepath):
        logger.info("Already analyzed, re-queuing for plot: %s", Path(filepath).name)
        radar_id = self.config.downloader.radar_id
        if self.output_queue:
            from adapt.setup_directories import get_analysis_path
            from datetime import datetime, timezone
            try:
                parts = file_id.split('_')
                datetime_str = parts[0][-8:] + parts[1]
                scan_time = datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
            except:
                scan_time = datetime.now(timezone.utc)
            seg_nc_path = get_analysis_path(
                output_dirs=self.output_dirs,
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

    def _load_and_regrid(self, filepath):
        """Load and regrid radar file, return ds, ds_2d, nc_full_path, scan_time."""
        save_netcdf = self.config.regridder.save_netcdf
        nc_full_path = None
        scan_time = None
        
        from adapt.setup_directories import get_netcdf_path
        from datetime import datetime, timezone
        radar_id = self.config.downloader.radar_id
        nc_filename = Path(filepath).stem
        try:
            parts = nc_filename.split('_')
            datetime_str = parts[0][-8:] + parts[1]
            scan_time = datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
        except:
            scan_time = datetime.now(timezone.utc)
        nc_full_path = get_netcdf_path(self.output_dirs, radar_id, nc_filename, scan_time=scan_time)
        output_dir = str(nc_full_path.parent)
        
        ds = None
        if nc_full_path and nc_full_path.exists() and nc_full_path.stat().st_size > 1000:
            try:
                import pyart
                grid = pyart.io.read_grid(str(nc_full_path))
                ds = grid.to_xarray()
                logger.debug("Loaded existing NetCDF: %s", nc_full_path.name)
            except Exception as e:
                logger.warning("Failed to load existing NetCDF %s: %s", nc_full_path.name, e)
                ds = None
        if ds is None:
            ds = self.loader.load_and_regrid(filepath, grid_kwargs=None,
                                            save_netcdf=save_netcdf, output_dir=output_dir)
        ds_2d = self._extract_2d_slice(ds)
        logger.debug(f"Extracted 2D slice at z-level, shape: {ds_2d.dims}")
        return ds, ds_2d, nc_full_path, scan_time

    def _compute_projections(self, ds_2d: xr.Dataset, filepath: str) -> xr.Dataset:
        """Compute optical flow projections if 2+ frames available.
        
        Parameters
        ----------
        ds_2d : xr.Dataset
            Segmented 2D dataset
        filepath : str
            Current file path (for history tracking)
            
        Returns
        -------
        xr.Dataset
            Dataset with projections added (flow_u, flow_v, cell_projections),
            or original ds_2d if insufficient frames
        """
        # Update history with current dataset
        self.dataset_history.append((filepath, ds_2d))
        if len(self.dataset_history) > self.max_history:
            self.dataset_history.pop(0)
        
        # Need 2+ frames for projection
        if len(self.dataset_history) < 2:
            logger.debug("First frame: optical flow not available (need 2+ datasets)")
            return ds_2d
        
        try:
            ds_prev_path, ds_prev = self.dataset_history[-2]
            ds_curr_path, ds_curr = self.dataset_history[-1]
            
            # Projector returns 2D ds with cell_projections, flow_u, flow_v added
            ds_with_proj = self.projector.project([ds_prev, ds_curr])
            if ds_with_proj is not None:
                logger.info(f"Projections computed: {list(ds_with_proj.data_vars)}")
                return ds_with_proj
        except Exception as e:
            logger.error(f"Optical flow projection failed: {e}", exc_info=True)
        
        return ds_2d

    def _get_table_columns(self) -> list:
        """Get list of column names from existing cells table."""
        try:
            cursor = self.db_conn.execute("PRAGMA table_info(cells)")
            return [row[1] for row in cursor.fetchall()]
        except:
            return []

    def _align_dataframe_to_schema(self, df_cells: pd.DataFrame) -> pd.DataFrame:
        """Align DataFrame columns to existing table schema.
        
        Ensures all columns from the table schema are present in the DataFrame,
        filling missing columns with NULL values. This handles cases where:
        - First file creates schema with only available columns
        - Subsequent files have new columns from enhanced extraction
        - Early frames lack projection data (need 2+ frames)
        """
        existing_cols = self._get_table_columns()
        if existing_cols:
            # Add missing columns with NULL values
            missing_cols = [c for c in existing_cols if c not in df_cells.columns]
            if missing_cols:
                logger.debug("DataFrame missing %d columns: %s (filling with NULL)", len(missing_cols), missing_cols[:5])
                for col in missing_cols:
                    df_cells[col] = None
            
            # Ensure column order matches table schema
            df_cells = df_cells[existing_cols]
        return df_cells

    def _analyze_and_save(self, ds_2d, filepath, nc_full_path, num_cells, scan_time):
        """Analyze cells (now has projection info), update tracker, insert into SQLite DB."""
        z_level = self.config.global_.z_level
        df_cells = self.analyzer.extract(ds_2d, z_level=z_level)
        
        # Contract: analysis output must meet requirements
        assert_analysis_output(df_cells)
        
        file_id = Path(filepath).stem
        if df_cells.empty:
            logger.debug("No cells detected in: %s", Path(filepath).name)
            return True

        # Ensure time columns are datetime for SQLite TIMESTAMP
        time_cols = ["time_volume_start", "time", "time_volume_end"]
        for col in time_cols:
            if col in df_cells.columns:
                df_cells[col] = pd.to_datetime(df_cells[col], errors='coerce')

        logger.debug("Analyzed: %d cells with projection info", len(df_cells))
        self._log_cell_statistics(df_cells)
        
        with self.output_lock:
            if not self.db_initialized:
                self._create_cells_table(df_cells)
                self.db_initialized = True
            else:
                # Align DataFrame to existing table schema (handles files with different variables)
                df_cells = self._align_dataframe_to_schema(df_cells)
            # Append data (schema already exists)
            df_cells.to_sql('cells', self.db_conn, if_exists='append', index=False)
        
        logger.info("Successfully processed: %s (saved %d cells to DB)", Path(filepath).name, len(df_cells))
        tracker = self.file_tracker
        if tracker:
            tracker.mark_stage_complete(file_id, "analyzed", num_cells=len(df_cells))
        return True

    def run(self):
        """Main processor loop (runs in thread).

        Continuously reads filepaths from input_queue, processes each file through
        the complete scientific pipeline, and sends results to output_queue for
        visualization. Exits when None (sentinel) is received.

        Notes
        -----
        Called automatically by thread.start(). Do not call directly.
        """
        logger.info("Processor started, waiting for files...")

        while not self.stopped():
            try:
                # Get next file from queue (block for 1 second)
                try:
                    filepath = self.input_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # Process file (always mark as done even if it fails)
                try:
                    success = self.process_file(filepath)
                except Exception as e:
                    logger.exception("Failed to process file: %s", filepath)
                finally:
                    # Always mark task as done to prevent queue from blocking
                    self.input_queue.task_done()

            except Exception as e:
                logger.exception("Processor error")

        logger.info("Processor stopped")


    def get_results(self) -> pd.DataFrame:
        """Return all processed cell statistics as a DataFrame.

        Thread-safe query of the SQLite database containing all cells extracted
        from processed files. Can be called while processor is running (WAL mode
        enables concurrent access).

        Returns
        -------
        pd.DataFrame
            One row per detected cell with columns for time, geometry, reflectivity,
            motion, and projections. Empty DataFrame if no processing completed yet.
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
        """Export all cell statistics to Parquet file.

        Creates a single Parquet file containing all cells from the SQLite
        database. Parquet format is efficient for subsequent analysis with Pandas,
        DuckDB, Polars, or other analytical tools.

        Parameters
        ----------
        filepath : str, optional
            Output Parquet filepath. If None, uses:
            `{output_dirs['analysis']}/{radar_id}_cells_statistics.parquet`
        """
        if filepath is None:
            analysis_dir = Path(self.output_dirs["analysis"])
            filepath = analysis_dir / f"{self.config.downloader.radar_id}_cells_statistics.parquet"

        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with self.output_lock:
                cursor = self.db_conn.execute("SELECT COUNT(*) FROM cells")
                count = cursor.fetchone()[0]

                if count > 0:
                    df = pd.read_sql("SELECT * FROM cells", self.db_conn)
                    df.to_parquet(filepath, engine='pyarrow',
                                  compression=self.config.output.compression,
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
                        f"Area [km2] - min={area_vals.min():.1f}, "
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
            logger.info("Cell Statistics: %s", summary_msg)

            # Log individual cell details for small number of cells
            if num_cells <= 5:
                for idx, row in df_cells.iterrows():
                    cell_label = row["cell_label"]
                    area = row["cell_area_sqkm"]
                    refl_mean = row["radar_reflectivity_mean"]
                    refl_max = row["radar_reflectivity_max"]

                    detail_msg = f"Cell #{cell_label}: area={area:.1f} km2, refl_mean={refl_mean:.1f} dBZ, refl_max={refl_max:.1f} dBZ"
                    logger.info("  %s", detail_msg)

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
        z_level = self.config.global_.z_level
        time_name = self.config.global_.coord_names.time
        z_name = self.config.global_.coord_names.z
        
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

    def _save_segmentation_netcdf(self, ds: xr.Dataset, filepath: str, scan_time) -> Optional[str]:
        """Save analysis results to NetCDF for visualization.
        """
        try:
            from adapt.setup_directories import get_analysis_path
            radar_id = self.config.downloader.radar_id

            # Create output path
            filename_stem = Path(filepath).stem
            seg_nc_filename = f"{filename_stem}_analysis.nc"
            seg_nc_path = get_analysis_path(
                self.output_dirs,
                radar_id=radar_id,
                analysis_type="netcdf",
                timestamp=scan_time,
                filename=seg_nc_filename
            )
            seg_nc_path.parent.mkdir(parents=True, exist_ok=True)

            # Add metadata
            ds.attrs.update({
                "source": str(filepath),
                "radar_id": radar_id,
                "description": "Radar analysis with segmentation and projections"
            })

            # Save to NetCDF
            ds.to_netcdf(seg_nc_path, mode='w', engine='netcdf4', format='NETCDF4', compute=True)
            ds.close()

            # Log what was saved
            components = list(ds.data_vars.keys())
            logger.info(f"Analysis saved: {seg_nc_path.name} [{', '.join(components)}]")

            return str(seg_nc_path)

        except Exception as e:
            logger.warning(f"Could not save segmentation NetCDF: {e}")
            return None


if __name__ == "__main__":
    # Logging configured by pipeline orchestrator
    pass  # Placeholder for demo code below
    print("Processor classes loaded. Use in pipeline orchestration.")
