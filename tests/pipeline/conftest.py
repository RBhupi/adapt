import pytest
import queue
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from adapt.pipeline.file_tracker import FileProcessingTracker


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def tracker(temp_dir):
    db_path = temp_dir / "tracker.db"
    return FileProcessingTracker(db_path)


@pytest.fixture
def basic_config(temp_dir):
    return {
        "mode": "historical",
        "global": {
            "z_level": 2000,
            "coord_names": {"time": "time", "z": "z"},
            "var_names": {"cell_labels": "cell_labels"},
        },
        "downloader": {
            "radar_id": "TEST",
        },
        "output_dirs": {
            "analysis": str(temp_dir / "analysis"),
            "logs": str(temp_dir / "logs"),
        },
        "segmenter": {},
        "projector": {},
        "analyzer": {},
    }


@pytest.fixture
def processor_queues():
    return queue.Queue(), queue.Queue()
