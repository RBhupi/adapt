from pathlib import Path
import pytest
from adapt.setup_directories import setup_output_directories

pytestmark = pytest.mark.unit


def test_setup_output_directories_creates_all(tmp_path):
    """Test that setup_output_directories creates base and logs dirs."""
    dirs = setup_output_directories(tmp_path)

    # New structure: only base and logs at root level
    expected = {"base", "logs"}

    assert set(dirs.keys()) == expected

    for path in dirs.values():
        assert isinstance(path, Path)
        assert path.exists()
        assert path.is_dir()


def test_setup_output_directories_is_idempotent(tmp_path):
    dirs1 = setup_output_directories(tmp_path)
    dirs2 = setup_output_directories(tmp_path)

    assert dirs1 == dirs2


def test_base_and_log_dirs_exist(tmp_path):
    """Test that base and logs directories exist after setup."""
    dirs = setup_output_directories(tmp_path)

    assert dirs["base"].exists()
    assert dirs["logs"].exists()
