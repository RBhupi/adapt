"""Pipeline Orchestrator: Coordinates Downloader → Processor → Plotter.

Manages the multi-threaded radar processing pipeline.

Author: Bhupendra Raut
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

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Manages the three-thread pipeline: Downloader → Queue → Processor.

    Responsibilities:
    - Initialize and start downloader and processor threads
    - Monitor queue depth and thread health
    - Handle graceful shutdown
    - Logging and error recovery
    """

    def __init__(self, config: dict, max_queue_size: int = 100):
        """Initialize orchestrator.

        Parameters
        ----------
        config : dict
            Full pipeline configuration (internal format).
        max_queue_size : int
            Maximum queue size before backpressure.
        """
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
        """Configure logging for the pipeline."""
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
        """Start the pipeline.

        Parameters
        ----------
        max_runtime : int, optional
            Maximum runtime in minutes. None = run until interrupted.
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

        logger.info("⏳ Grace period (10s)...")
        time.sleep(10)
        logger.info("✅ Historical mode complete")
        return True

    def _drain_queue(self, q: queue.Queue, name: str, timeout: int = 300):
        """Wait for queue to drain with timeout."""
        wait_count = 0
        while q.qsize() > 0:
            logger.info("⏳ Waiting for %s queue: %d remaining", name, q.qsize())
            time.sleep(5)
            wait_count += 1
            if wait_count > timeout // 5:
                logger.warning("%s queue drain timeout", name)
                break

    def stop(self):
        """Stop the pipeline gracefully."""
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
