"""AWS NEXRAD Level-II Downloader.

Downloads radar scans from AWS in realtime or historical mode.

Author: Sid Gupta and Bhupendra Raut
"""

import threading
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

from nexradaws import NexradAwsInterface

logger = logging.getLogger(__name__)


class AwsNexradDownloader(threading.Thread):
    """AWS NEXRAD downloader with realtime and historical modes.

    Realtime mode: Rolling window of latest files.
    Historical mode: All files in time range.
    """

    def __init__(self, config: dict, result_queue=None):
        """Initialize downloader.

        Parameters
        ----------
        config : dict
            - radar_id: NEXRAD radar ID (e.g., "KDIX")
            - output_dir: Where to save files
            - sleep_interval: Seconds between polls (realtime)
            - latest_n: Number of files to keep (realtime)
            - minutes: Time window in minutes (realtime)
            - start_time: ISO timestamp (historical)
            - end_time: ISO timestamp (historical)
        result_queue : Queue, optional
            Queue to notify downstream of new files.
        """
        super().__init__(daemon=True)

        self.config = config
        self.radar_id = config.get("radar_id", "KLOT")
        self.output_dir = Path(config.get("output_dir", "./data"))
        self.sleep_interval = config.get("sleep_interval", 30)
        self.latest_n = config.get("latest_n", 3)
        self.minutes = config.get("minutes", 30)
        self.start_time = config.get("start_time", None)
        self.end_time = config.get("end_time", None)

        self.result_queue = result_queue
        self.conn = NexradAwsInterface()

        self._stop_event = threading.Event()
        self._known_files = set()
        self._known_files_lock = threading.Lock()
        self._min_file_size = 1024  # bytes

        # Historical mode tracking
        self._historical_complete = threading.Event()
        self._expected_scans = 0
        self._processed_scans = 0

        self.name = f"Downloader-{self.radar_id}"

    # ========================================================================
    # Thread control
    # ========================================================================

    def stop(self):
        """Signal thread to stop."""
        self._stop_event.set()

    def stopped(self) -> bool:
        """Check if stop was requested."""
        return self._stop_event.is_set()

    def is_historical_mode(self) -> bool:
        """Check if running in historical mode."""
        return bool(self.start_time and self.end_time)

    def is_historical_complete(self) -> bool:
        """Check if historical download is complete."""
        return self._historical_complete.is_set()

    def get_historical_progress(self) -> tuple:
        """Get (processed, expected) scan counts."""
        return self._processed_scans, self._expected_scans

    def run(self):
        """Main thread loop."""
        mode = "historical" if self.is_historical_mode() else "realtime"
        logger.info("Starting %s in %s mode", self.name, mode)

        while not self.stopped():
            try:
                self.download_task()
            except Exception as e:
                logger.exception("Download task failed")

            # Historical: exit after completion
            if self.is_historical_mode() and self.is_historical_complete():
                logger.info("✅ Historical download complete")
                break

            # Sleep between iterations (interruptible)
            self._interruptible_sleep(self.sleep_interval)

        logger.info("Stopped %s", self.name)

    def _interruptible_sleep(self, seconds: int):
        """Sleep that can be interrupted by stop event."""
        for _ in range(seconds // 2):
            if self.stopped():
                break
            if self.is_historical_mode() and self.is_historical_complete():
                break
            time.sleep(2)

    # ========================================================================
    # Download task - dispatches to realtime or historical
    # ========================================================================

    def download_task(self) -> list:
        """Main download task. Returns list of new downloads."""
        if self.is_historical_mode():
            return self._download_historical()
        else:
            return self._download_realtime()

    def _download_historical(self) -> list:
        """Download files for historical time range."""
        start, end = self._parse_time_range()
        logger.info("Historical: %s to %s", start, end)

        scans = self._fetch_scans(start, end)
        if not scans:
            self._historical_complete.set()
            return []

        self._expected_scans = len(scans)
        return self._process_scans(scans)

    def _download_realtime(self) -> list:
        """Download latest files for realtime mode."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=self.minutes)
        logger.info("Realtime: last %d min (%s to %s)", self.minutes, start, end)

        scans = self._fetch_scans(start, end)
        if not scans:
            return []

        # Keep only latest N
        scans = scans[-self.latest_n:]
        logger.info("Keeping latest %d scans", len(scans))

        return self._process_scans(scans)

    # ========================================================================
    # Scan fetching and processing
    # ========================================================================

    def _parse_time_range(self) -> tuple:
        """Parse ISO timestamps to datetime objects."""
        start = datetime.fromisoformat(self.start_time.replace("Z", "+00:00"))
        end = datetime.fromisoformat(self.end_time.replace("Z", "+00:00"))
        return start, end

    def _fetch_scans(self, start: datetime, end: datetime) -> list:
        """Fetch available scans from AWS."""
        try:
            scans = self.conn.get_avail_scans_in_range(start, end, self.radar_id)
            scans = sorted(scans, key=lambda s: s.scan_time)

            # Filter out MDM files
            scans = [s for s in scans if not s.key.endswith("_MDM")]

            logger.info("Found %d scans for %s", len(scans), self.radar_id)
            return scans
        except Exception as e:
            logger.error("Failed to fetch scans: %s", e)
            return []

    def _process_scans(self, scans: list) -> list:
        """Process list of scans: download if needed, queue only if file exists."""
        new_downloads = []
        queued = 0

        for scan in scans:
            if self.stopped():
                break

            local_path = self._get_local_path(scan)
            is_new = False

            # Download if not exists
            if not self._file_exists(local_path):
                if self._download_scan(scan, local_path):
                    is_new = True
                    new_downloads.append(local_path)
                else:
                    logger.warning("Failed to download: %s", scan.key)
                    continue  # Skip queueing if download failed

            # Queue the file ONLY if it now exists on disk
            if self._file_exists(local_path):
                with self._known_files_lock:
                    if local_path not in self._known_files:
                        self._notify_queue(local_path, scan.scan_time, is_new)
                        self._known_files.add(local_path)
                        queued += 1
            else:
                logger.error("File missing after download attempt: %s", local_path)

        logger.info("Processed: %d queued, %d new downloads", queued, len(new_downloads))

        # Mark historical complete if all processed
        if self.is_historical_mode():
            self._processed_scans = queued
            if queued >= len(scans):
                self._historical_complete.set()

        return new_downloads

    def _get_local_path(self, scan) -> Path:
        """Get local path for scan (YYYYMMDD/RADAR_ID/filename)."""
        date_str = scan.scan_time.strftime("%Y%m%d")
        filename = Path(scan.key).name
        return (self.output_dir / date_str / self.radar_id / filename).resolve()

    def _file_exists(self, path: Path) -> bool:
        """Check if valid file exists."""
        try:
            return path.exists() and path.stat().st_size >= self._min_file_size
        except:
            return False

    def _download_scan(self, scan, local_path: Path) -> bool:
        """Download single scan to local path."""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download to temp then move
            temp_dir = self.output_dir / "_temp"
            temp_dir.mkdir(exist_ok=True)

            results = self.conn.download([scan], temp_dir, keep_aws_folders=False)
            success = list(results.iter_success())

            if success:
                temp_file = Path(success[0].filepath)
                if temp_file.exists():
                    temp_file.rename(local_path)
                    logger.info("✓ Downloaded: %s", local_path.name)
                    return True

            return False
        except Exception as e:
            logger.error("Download failed: %s - %s", scan.key, e)
            return False

    def _notify_queue(self, path: Path, scan_time: datetime, is_new: bool):
        """Put file notification in result queue."""
        if self.result_queue is None:
            return

        try:
            # Register with tracker if available
            tracker = self.config.get("file_tracker")
            file_id = path.stem
            if tracker:
                tracker.register_file(file_id, self.radar_id, scan_time, path)
                tracker.mark_stage_complete(file_id, "downloaded", path=path)

            self.result_queue.put({
                "path": path,
                "scan_time": scan_time,
                "radar_id": self.radar_id,
                "file_id": file_id,
            })
        except Exception as e:
            logger.error("Failed to queue notification: %s", e)
