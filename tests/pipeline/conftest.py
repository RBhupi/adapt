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
    config = {
    "mode": "realtime",
    
    "reader": {
        "file_format": "nexrad_archive",
    },
    "regridder": {
        "grid_shape": (41, 301, 301),
        "grid_limits": ((0, 20000), (-150000, 150000), (-150000, 150000)),
        "roi_func": "dist_beam",
        "min_radius": 1750.0,
        "weighting_function": "cressman",
        "save_netcdf": True,
    },
    "logging": {
        "level": "INFO",
    },
}

    return config


# made for processor tests, very
@pytest.fixture
def processor_queues():
    return queue.Queue(), queue.Queue()


