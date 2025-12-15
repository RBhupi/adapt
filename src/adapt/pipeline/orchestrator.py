"""Multi-threaded pipeline orchestration.

Coordinates downloader, processor, and plotter threads with queue-based
inter-thread communication. Manages lifecycle, monitoring, and graceful shutdown.
"""

import queue
import time
import logging
from pathlib import Path
from typing import Optional

from adapt.radar.downloader import AwsNexradDownloader
from adapt.pipeline.processor import RadarProcessor
from adapt.pipeline.file_tracker import FileProcessingTracker
from adapt.visualization.plotter import PlotterThread

__all__ = ['PipelineOrchestrator']

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Manages the multi-threaded radar processing pipeline.

    This is the main entry point for running ``adapt``. It coordinates three
    worker threads (downloader, processor, plotter) using queues for
    inter-thread communication. The orchestrator handles startup, monitoring,
    and graceful shutdown of the entire pipeline.

    **Pipeline Architecture:**

    1. **Downloader Thread**: Discovers and downloads NEXRAD Level-II files
       from AWS in realtime or historical mode.

    2. **Processor Thread**: Processes downloaded files through the full
       scientific pipeline:
       - Load and regrid Level-II data to Cartesian grid
       - Segment cells using configurable threshold method
       - Compute motion projections (frame 2 and beyond)
       - Extract cell-level statistics
       - Persist segmentation to NetCDF and statistics to SQLite

    3. **Plotter Thread**: Generates visualization PNG files from analysis
       results (reflectivity + segmentation + projections).

    **Modes:**

    - **Realtime**: Polls for latest files within a rolling time window
      (e.g., "files from last 60 minutes"). Useful for operational monitoring.

    - **Historical**: Downloads all files within a fixed time range
      (start_time to end_time). Useful for batch reprocessing and research.

    **Queue Management:**

    Inter-thread queues have configurable size limits (default 100 items).
    Larger queues enable higher throughput but use more memory. Smaller
    queues provide backpressure (slow down downloader if processor falls behind).

    **File Tracking:**

    The FileProcessingTracker SQLite database records the state of each file
    (downloaded, regridded, analyzed, plotted, or failed). This enables:
    - Resumable processing (restart without reprocessing completed files)
    - Progress tracking
    - Failure recovery and debugging

    **Logging:**

    All output goes to both console and log file (logs/{radar_id}_pipeline.log).
    Log level controlled via config: "DEBUG", "INFO", "WARNING", "ERROR".

    Example usage::

        from adapt.pipeline.orchestrator import PipelineOrchestrator
        
        config = {
            "mode": "realtime",
            "downloader": {"radar_id": "KDIX", ...},
            "output_dirs": {...},
            ...
        }
        
        orch = PipelineOrchestrator(config)
        orch.start(max_runtime=60)  # Run for 60 minutes then stop
    """

    def __init__(self, config: dict, max_queue_size: int = 100):
        """Initialize orchestrator with pipeline configuration.

        Parameters
        ----------
        config : dict
            Complete pipeline configuration dictionary containing:
            
            - `mode` : str, "realtime" or "historical"
                Processing mode. Realtime continuously monitors for new files.
                Historical downloads all files within a time range then stops.
            
            - `base_dir` : str
                Root output directory. Subdirectories created for nexrad/,
                gridded/, analysis/, plots/, logs/.
            
            - `output_dirs` : dict
                Pre-computed output directory paths. Created by
                `setup_output_directories()`.
            
            - `downloader` : dict
                Download configuration:
                
                - `radar_id` : str, e.g. "KDIX"
                - `output_dir` : str, directory to save Level-II files
                - `latest_n` : int, realtime mode only. Keep last N files.
                - `minutes` : int, realtime mode. Rolling window in minutes.
                - `sleep_interval` : int, seconds between polls.
                - `start_time` : str, historical mode. ISO timestamp.
                - `end_time` : str, historical mode. ISO timestamp.
            
            - `regridder`, `segmenter`, `analyzer`, `projector`, `visualization` : dict
                Configuration for each processing stage (passed to worker threads).
            
            - `logging` : dict
                - `level` : str, "DEBUG", "INFO", "WARNING", "ERROR"
            
            - `file_tracker` : FileProcessingTracker, optional
                If provided, used to track file processing state. Created
                internally if not provided.

        max_queue_size : int, optional
            Maximum size of inter-thread communication queues (default: 100).
            
            - Larger queues (200+): Higher throughput, more memory, less backpressure
            - Smaller queues (10-30): Lower memory, stronger backpressure, risk of stalls
            - Balance depends on your file processing speed vs download speed
        """
        # TODO: We will use pydantic for config validation later.
        self.config = config
        self.max_queue_size = max_queue_size

        # Queues for inter-thread communication
        self.downloader_queue = queue.Queue(maxsize=max_queue_size)
        self.plotter_queue = queue.Queue(maxsize=50)

        # Threads (created in start())
        self.downloader = None
        self.processor = None
        self.plotter = None

        # File tracking
        self.tracker = None

        # Lifecycle state
        self._stop_event = False
        self._start_time = None
        self._max_duration = None

    def _setup_logging(self):
        """Configure logging and file tracking systems.

        Initializes root logger with file and console handlers, creates output
        directories, and sets up FileProcessingTracker for pipeline state management.
        Log level and paths derived from config.
        """
        #> loging level can be one of the config_parameters tat will decide if the loging is done
        #> frequently or only for important events, that is debug or info or only warnings and errors.
        log_level = self.config.get("logging", {}).get("level", "INFO")
        log_level = getattr(logging, log_level.upper(), logging.INFO)

        # Get log path from output_dirs.
        output_dirs = self.config.get("output_dirs", {})
        radar_id = self.config.get("downloader", {}).get("radar_id", "UNKNOWN")
        log_dir = Path(output_dirs.get("logs", "."))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"pipeline_{radar_id}.log"

        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Clear existing handlers and add new ones
        root = logging.getLogger()
        root.setLevel(log_level)
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        # File handler
        fh = logging.FileHandler(log_path)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        root.addHandler(ch)

        logger.info("Logging: level=%s, file=%s", log_level, log_path)

        # Initialize file tracker
        if output_dirs:
            tracker_path = Path(output_dirs.get("analysis", ".")) / f"{radar_id}_file_tracker.db"
            self.tracker = FileProcessingTracker(tracker_path)
            self.config["file_tracker"] = self.tracker
            if "downloader" in self.config:
                self.config["downloader"]["file_tracker"] = self.tracker
            logger.info("File tracker: %s", tracker_path)

    def start(self, max_runtime: Optional[int] = None):
        """Start the pipeline and run until completion or user interrupt.

        This is a blocking call that starts the downloader, processor, and plotter
        threads, then enters a monitoring loop. The loop logs status every 30 seconds
        and handles mode-specific exit conditions.

        **Realtime Mode:** Runs until you press Ctrl+C or max_runtime is exceeded.

        **Historical Mode:** Automatically exits when all files within the
        start_time/end_time range are queued, processed, and plotted.

        All output (console + file) logged to logs/{radar_id}_pipeline.log
        at the level specified in config["logging"]["level"].

        Parameters
        ----------
        max_runtime : int, optional
            Maximum runtime in minutes (realtime mode only).
            If None, runs until KeyboardInterrupt (Ctrl+C).
            Ignored in historical mode (uses file completion instead).

        Raises
        ------
        KeyboardInterrupt
            User pressed Ctrl+C. Pipeline stops gracefully.

        Notes
        -----
        To stop the pipeline gracefully, press Ctrl+C. Do NOT force-kill.
        The stop() method is called automatically to close threads and save results.

        Examples
        --------
        Run for 60 minutes in realtime mode::

            orch = PipelineOrchestrator(config)
            orch.start(max_runtime=60)

        Run historical mode until all files processed::

            orch = PipelineOrchestrator(config)  # mode="historical" in config
            orch.start()  # Runs until all files between start_time/end_time processed
        """
        self._setup_logging()

        logger.info("=" * 60)
        logger.info("Starting Radar Processing Pipeline")
        logger.info("=" * 60)

        self._start_time = time.time()
        self._max_duration = max_runtime * 60 if max_runtime else None

        if self._max_duration:
            logger.info("Max runtime: %d minutes", max_runtime)
        else:
            logger.info("Max runtime: Until interrupted")

        # Start Downloader thread
        logger.info("Starting Downloader...")
        self.downloader = AwsNexradDownloader(
            config=self.config.get("downloader", {}),
            result_queue=self.downloader_queue
        )
        self.downloader.start()
        logger.info("✓ Downloader started")

        # Start Processor thread
        logger.info("Starting Processor...")
        self.processor = RadarProcessor(
            input_queue=self.downloader_queue,
            config=self.config,
            output_queue=self.plotter_queue,
        )
        self.processor.start()
        logger.info("✓ Processor started")

        # Start Plotter thread
        output_dirs = self.config.get("output_dirs", {})
        logger.info("Starting Plotter...")
        self.plotter = PlotterThread(
            input_queue=self.plotter_queue,
            output_dirs=output_dirs,
            config=self.config,
            name="RadarPlotter"
        )
        self.plotter.start()
        logger.info("✓ Plotter started")

        mode = self.config.get("mode", "realtime")
        logger.info("Pipeline running in %s mode. Press Ctrl+C to stop.", mode.upper())

        try:
            self._main_loop(mode)
        except KeyboardInterrupt:
            logger.info("\nShutdown signal received (Ctrl+C)")
        finally:
            self.stop()

    def _main_loop(self, mode: str):
        """Main monitoring loop."""
        while True:
            # Historical mode: check for completion
            if mode == "historical":
                if self._check_historical_complete():
                    break

            # Realtime mode: check duration limit
            if mode == "realtime" and self._max_duration:
                elapsed = time.time() - self._start_time
                if elapsed > self._max_duration:
                    logger.info("Max duration reached")
                    break

            # Status every 30 seconds
            time.sleep(30)
            self._log_status()

    def _check_historical_complete(self) -> bool:
        """Check if historical mode is complete. Returns True to exit."""
        downloader_complete = (
            self.downloader.is_historical_complete() or
            not self.downloader.is_alive()
        )

        if not downloader_complete:
            return False

        processed, expected = self.downloader.get_historical_progress()
        logger.info("📦 Downloader complete: %d/%d files queued", processed, expected)

        # Stop downloader explicitly to signal thread termination
        self.downloader.stop()
        logger.info("Stopping downloader thread...")
        self.downloader.join(timeout=5)
        if self.downloader.is_alive():
            logger.warning("Downloader thread did not stop cleanly")

        # Wait for queues to drain
        self._drain_queue(self.downloader_queue, "processor")
        self._drain_queue(self.plotter_queue, "plotter")

        # Now stop processor and plotter threads
        logger.info("Stopping processor and plotter threads...")
        if self.processor and self.processor.is_alive():
            self.processor.stop()
            self.processor.join(timeout=10)
            if self.processor.is_alive():
                logger.warning("Processor thread did not stop cleanly")
        
        if self.plotter and self.plotter.is_alive():
            self.plotter.stop()
            self.plotter.join(timeout=10)
            if self.plotter.is_alive():
                logger.warning("Plotter thread did not stop cleanly")

        logger.info("✅ Historical mode complete")
        return True

    def _drain_queue(self, q: queue.Queue, name: str, timeout: int = 300):
        """Wait for queue to drain with timeout."""
        wait_count = 0
        start_time = time.time()
        last_size = q.qsize()
        
        while q.qsize() > 0:
            current_size = q.qsize()
            if current_size == last_size:
                wait_count += 1
            else:
                wait_count = 0  # Reset if progress is being made
                last_size = current_size
            
            logger.info("⏳ Waiting for %s queue: %d remaining", name, current_size)
            time.sleep(5)
            
            # Check timeout both by iteration count and elapsed time
            if wait_count > timeout // 5 or (time.time() - start_time) > timeout:
                logger.warning("%s queue drain timeout (%d/%d seconds)", 
                              name, int(time.time() - start_time), timeout)
                break

    def stop(self):
        """Stop the pipeline gracefully and finalize all results.

        Called automatically when start() exits (either by user interrupt or
        mode-specific completion). Safe to call multiple times.

        **Operations:**

        1. Signals all worker threads to stop
        2. Waits up to 5 seconds for each thread to finish
        3. Saves accumulated cell statistics to SQLite database
        4. Generates final summary statistics (total cells, completion times)
        5. Closes database connections
        6. Logs pipeline runtime and file processing statistics

        Notes
        -----
        This method should complete within ~20 seconds. If a thread does not
        respond to the stop signal within its timeout, a warning is logged but
        the shutdown continues (graceful degradation).

        Files processed before the final save() will have their statistics
        in the SQLite database. The database is safe to query even while the
        pipeline is running (uses WAL mode).
        """
        if self._stop_event:
            return

        self._stop_event = True
        logger.info("Stopping pipeline...")

        # Stop threads
        for name, thread in [("Downloader", self.downloader),
                              ("Processor", self.processor),
                              ("Plotter", self.plotter)]:
            if thread and thread.is_alive():
                logger.info("Stopping %s...", name)
                thread.stop()
                thread.join(timeout=5)
                if thread.is_alive():
                    logger.warning("%s did not stop cleanly", name)

        # Save results
        if self.processor:
            logger.info("Saving results...")
            self.processor.save_results()
            df = self.processor.get_results()
            logger.info("Final results: %d rows", len(df))
            self.processor.close_database()

        # Summary
        elapsed = time.time() - self._start_time if self._start_time else 0
        logger.info("=" * 60)
        logger.info("Pipeline stopped. Runtime: %.1f seconds", elapsed)

        if self.tracker:
            stats = self.tracker.get_statistics()
            total_cells = stats.get('total_cells', 0)
            if total_cells is None:
                total_cells = 0
            logger.info("Statistics: total=%d, completed=%d, cells=%d",
                       stats.get('total', 0), stats.get('completed', 0), total_cells)
            self.tracker.close()

        logger.info("=" * 60)

    def _log_status(self):
        """Log current pipeline status."""
        mode = self.config.get("mode", "realtime")
        hist_status = ""
        if mode == "historical" and self.downloader:
            processed, expected = self.downloader.get_historical_progress()
            hist_status = f" [{processed}/{expected}]"

        logger.info(
            "Status: D=%s P=%s L=%s Q=%d PQ=%d%s",
            "✓" if self.downloader and self.downloader.is_alive() else "✗",
            "✓" if self.processor and self.processor.is_alive() else "✗",
            "✓" if self.plotter and self.plotter.is_alive() else "✗",
            self.downloader_queue.qsize(),
            self.plotter_queue.qsize(),
            hist_status
        )
