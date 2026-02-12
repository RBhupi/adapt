import pytest
import queue
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from adapt.pipeline.file_tracker import FileProcessingTracker
from adapt.schemas import ParamConfig, InternalConfig, UserConfig
from adapt.schemas.resolve import resolve_config
from adapt.setup_directories import setup_output_directories


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
def pipeline_config(temp_dir) -> InternalConfig:
    """InternalConfig for pipeline tests."""
    param = ParamConfig()
    # For tests, provide defaults since radar_id and base_dir are required at runtime
    user = UserConfig(
        radar_id="TEST_RADAR",
        base_dir=str(temp_dir)
    )
    return resolve_config(param, user, None)


@pytest.fixture
def pipeline_output_dirs(temp_dir):
    """Output directories for pipeline tests.

    Returns dict with 'base' and 'logs' from setup_output_directories,
    plus backward-compatible keys that point to base for legacy tests.
    """
    dirs = setup_output_directories(temp_dir)
    # Add legacy keys for backward compatibility in tests
    dirs["nexrad"] = dirs["base"]
    dirs["gridnc"] = dirs["base"]
    dirs["analysis"] = dirs["base"]
    dirs["plots"] = dirs["base"]
    return dirs


# made for processor tests
@pytest.fixture
def processor_queues():
    return queue.Queue(), queue.Queue()


