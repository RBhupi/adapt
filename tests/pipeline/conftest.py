import pytest
import queue
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from adapt.pipeline.file_tracker import FileProcessingTracker
from adapt.schemas.param import ParamConfig
from adapt.schemas.internal import InternalConfig  
from adapt.schemas.user import UserConfig
from adapt.schemas.resolve import resolve_config
from adapt.setup_directories import setup_output_directories
from adapt.core import DataRepository


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
    config_dict = resolve_config(param, user, None).model_dump()
    
    # Add required fields for new architecture
    output_dirs = setup_output_directories(str(temp_dir))
    config_dict["output_dirs"] = {k: str(v) for k, v in output_dirs.items()}
    config_dict["run_id"] = DataRepository.generate_run_id()
    
    return InternalConfig.model_validate(config_dict)


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


