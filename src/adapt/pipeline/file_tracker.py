"""SQLite-based file processing state tracker.

Tracks radar files through pipeline stages (downloaded, regridded, analyzed, plotted).
Enables idempotent processing with stop/restart, progress tracking, and failure recovery.
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, List
import threading

logger = logging.getLogger(__name__)


class FileProcessingTracker:
    """Tracks file state and lifecycle across pipeline stages.

    Records downloads, regridding, analysis, and visualization completion.
    Enables resumable processing: stopped jobs can restart without reprocessing
    completed files. Thread-safe via internal locking.
    """

    def __init__(self, db_path: Path | str):
        """Initialize tracker.

        Parameters
        ----------
        db_path : Path or str
            Path to SQLite database file. Created if doesn't exist.
            Typically: output_dirs/analysis/{radar_id}_file_tracker.db
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = None
        self._lock = threading.Lock()

        # Initialize database
        self._init_database()
        logger.info(f" File tracker initialized: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row  # Enable dict-like access
        return self._conn

    def _init_database(self):
        """Create database schema if it doesn't exist."""
        conn = self._get_connection()

        with self._lock:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_processing (
                    file_id TEXT PRIMARY KEY,
                    radar_id TEXT NOT NULL,
                    scan_time TEXT NOT NULL,

                    nexrad_path TEXT,
                    gridnc_path TEXT,
                    analysis_path TEXT,
                    plot_path TEXT,

                    downloaded_at TEXT,
                    regridded_at TEXT,
                    analyzed_at TEXT,
                    plotted_at TEXT,

                    status TEXT DEFAULT 'pending',
                    error_message TEXT,

                    file_size_mb REAL,
                    num_cells INTEGER,

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_radar_id ON file_processing(radar_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON file_processing(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scan_time ON file_processing(scan_time)")

            conn.commit()

    def register_file(self, file_id: str, radar_id: str, scan_time: datetime,
                     nexrad_path: Optional[Path] = None) -> bool:
        """
        Register a new file for tracking.

        Parameters
        ----------
        file_id : str
            Unique file identifier (filename without extension)
        radar_id : str
            Radar identifier (e.g., KHTX)
        scan_time : datetime
            Scan timestamp
        nexrad_path : Path, optional
            Path to original NEXRAD file

        Returns
        -------
        bool
            True if newly registered, False if already exists
        """
        conn = self._get_connection()

        with self._lock:
            # Check if already exists
            cursor = conn.execute("SELECT file_id FROM file_processing WHERE file_id = ?", (file_id,))
            if cursor.fetchone():
                return False

            # Calculate file size if path provided
            file_size_mb = None
            if nexrad_path and nexrad_path.exists():
                file_size_mb = nexrad_path.stat().st_size / (1024 * 1024)

            conn.execute("""
                INSERT INTO file_processing
                (file_id, radar_id, scan_time, nexrad_path, file_size_mb, downloaded_at, status)
                VALUES (?, ?, ?, ?, ?, ?, 'pending')
            """, (
                file_id,
                radar_id,
                scan_time.isoformat(),
                str(nexrad_path) if nexrad_path else None,
                file_size_mb,
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()

            logger.debug(f"Registered file: {file_id}")
            return True

    def mark_stage_complete(self, file_id: str, stage: str,
                          path: Optional[Path] = None,
                          num_cells: Optional[int] = None,
                          error: Optional[str] = None):
        """Mark a processing stage as complete.

        Parameters
        ----------
        file_id : str
            File identifier
        stage : str
            Stage name: 'downloaded', 'regridded', 'analyzed', 'plotted'
        path : Path, optional
            Path to output file for this stage
        num_cells : int, optional
            Number of cells (for analyzed stage)
        error : str, optional
            Error message if stage failed
        """
        valid_stages = ['downloaded', 'regridded', 'analyzed', 'plotted']
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}")

        conn = self._get_connection()
        timestamp_col = f"{stage}_at"

        # Map stage to path column
        stage_to_path = {
            'downloaded': 'nexrad_path',
            'regridded': 'gridnc_path',
            'analyzed': 'analysis_path',
            'plotted': 'plot_path'
        }
        path_col = stage_to_path[stage]

        with self._lock:
            # Determine new status
            if error:
                new_status = 'failed'
            elif stage == 'plotted':
                new_status = 'completed'
            else:
                new_status = 'processing'

            # Update record
            if num_cells is not None:
                conn.execute(f"""
                    UPDATE file_processing
                    SET {timestamp_col} = ?,
                        {path_col} = ?,
                        num_cells = ?,
                        status = ?,
                        error_message = ?,
                        updated_at = ?
                    WHERE file_id = ?
                """, (
                    datetime.now(timezone.utc).isoformat(),
                    str(path) if path else None,
                    num_cells,
                    new_status,
                    error,
                    datetime.now(timezone.utc).isoformat(),
                    file_id
                ))
            else:
                conn.execute(f"""
                    UPDATE file_processing
                    SET {timestamp_col} = ?,
                        {path_col} = ?,
                        status = ?,
                        error_message = ?,
                        updated_at = ?
                    WHERE file_id = ?
                """, (
                    datetime.now(timezone.utc).isoformat(),
                    str(path) if path else None,
                    new_status,
                    error,
                    datetime.now(timezone.utc).isoformat(),
                    file_id
                ))
            conn.commit()

            logger.debug(f"Marked {stage} complete: {file_id}")

    def get_file_status(self, file_id: str) -> Optional[Dict]:
        """Get processing status for a file.

        Parameters
        ----------
        file_id : str
            File identifier

        Returns
        -------
        dict or None
            File status dict or None if not found
        """
        conn = self._get_connection()

        with self._lock:
            cursor = conn.execute("""
                SELECT * FROM file_processing WHERE file_id = ?
            """, (file_id,))
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

    def get_pending_files(self, stage: Optional[str] = None,
                         radar_id: Optional[str] = None,
                         limit: Optional[int] = None) -> List[Dict]:
        """Get files that need processing.

        Parameters
        ----------
        stage : str, optional
            Filter by stage needing processing:
            - 'regridded': files downloaded but not regridded
            - 'analyzed': files regridded but not analyzed
            - 'plotted': files analyzed but not plotted
        radar_id : str, optional
            Filter by radar ID
        limit : int, optional
            Maximum number of files to return

        Returns
        -------
        list of dict
            List of file records
        """
        conn = self._get_connection()

        # Build query based on stage
        if stage == 'regridded':
            condition = "downloaded_at IS NOT NULL AND regridded_at IS NULL"
        elif stage == 'analyzed':
            condition = "regridded_at IS NOT NULL AND analyzed_at IS NULL"
        elif stage == 'plotted':
            condition = "analyzed_at IS NOT NULL AND plotted_at IS NULL"
        else:
            condition = "status != 'completed' AND status != 'failed'"

        query = f"SELECT * FROM file_processing WHERE {condition}"
        params = []

        if radar_id:
            query += " AND radar_id = ?"
            params.append(radar_id)

        query += " ORDER BY scan_time"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self._lock:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_statistics(self, radar_id: Optional[str] = None) -> Dict:
        """Get processing statistics.

        Parameters
        ----------
        radar_id : str, optional
            Filter by radar ID

        Returns
        -------
        dict
            Statistics dictionary
        """
        conn = self._get_connection()

        where_clause = f"WHERE radar_id = '{radar_id}'" if radar_id else ""

        with self._lock:
            cursor = conn.execute(f"""
                SELECT
                    COUNT(*) as total,
                    COUNT(downloaded_at) as downloaded,
                    COUNT(regridded_at) as regridded,
                    COUNT(analyzed_at) as analyzed,
                    COUNT(plotted_at) as plotted,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as processing,
                    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                    SUM(num_cells) as total_cells
                FROM file_processing
                {where_clause}
            """)
            row = cursor.fetchone()
            return dict(row) if row else {}

    def should_process(self, file_id: str, stage: str) -> bool:
        """Check if a file needs processing at given stage.

        Parameters
        ----------
        file_id : str
            File identifier
        stage : str
            Stage to check: 'regridded', 'analyzed', 'plotted'

        Returns
        -------
        bool
            True if file should be processed, False if already done
        """
        status = self.get_file_status(file_id)

        if not status:
            # File not registered, should process
            return True

        # Check if stage already completed
        timestamp_col = f"{stage}_at"
        return status.get(timestamp_col) is None

    def reset_failed(self, radar_id: Optional[str] = None):
        """Reset failed files to pending for retry.

        Parameters
        ----------
        radar_id : str, optional
            Filter by radar ID
        """
        conn = self._get_connection()

        with self._lock:
            if radar_id:
                conn.execute("""
                    UPDATE file_processing
                    SET status = 'pending', error_message = NULL, updated_at = ?
                    WHERE status = 'failed' AND radar_id = ?
                """, (datetime.now(timezone.utc).isoformat(), radar_id))
            else:
                conn.execute("""
                    UPDATE file_processing
                    SET status = 'pending', error_message = NULL, updated_at = ?
                    WHERE status = 'failed'
                """, (datetime.now(timezone.utc).isoformat(),))
            conn.commit()

            logger.info(f"Reset failed files to pending")

    def cleanup_deleted_files(self, radar_id: Optional[str] = None):
        """Remove tracking records for files that no longer exist on disk.

        Parameters
        ----------
        radar_id : str, optional
            Filter by radar ID
        """
        conn = self._get_connection()

        with self._lock:
            # Get all files
            where_clause = f"WHERE radar_id = '{radar_id}'" if radar_id else ""
            cursor = conn.execute(f"""
                SELECT file_id, nexrad_path FROM file_processing {where_clause}
            """)

            deleted = []
            for row in cursor.fetchall():
                file_id = row['file_id']
                nexrad_path = row['nexrad_path']

                if nexrad_path and not Path(nexrad_path).exists():
                    deleted.append(file_id)

            # Delete records
            if deleted:
                placeholders = ','.join('?' * len(deleted))
                conn.execute(f"""
                    DELETE FROM file_processing WHERE file_id IN ({placeholders})
                """, deleted)
                conn.commit()

                logger.info(f"Cleaned up {len(deleted)} deleted file(s)")

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


