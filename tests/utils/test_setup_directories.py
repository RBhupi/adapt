from pathlib import Path
from adapt.setup_directories import setup_output_directories


def test_setup_output_directories_creates_all(tmp_path):
    dirs = setup_output_directories(tmp_path)

    expected = {
        "base", "nexrad", "gridnc", "analysis", "plots", "logs"
    }

    assert set(dirs.keys()) == expected

    for path in dirs.values():
        assert isinstance(path, Path)
        assert path.exists()
        assert path.is_dir()

def test_setup_output_directories_is_idempotent(tmp_path):
    dirs1 = setup_output_directories(tmp_path)
    dirs2 = setup_output_directories(tmp_path)

    assert dirs1 == dirs2

def test_analysis_and_log_dirs_exist(tmp_path):
    dirs = setup_output_directories(tmp_path)

    assert (dirs["analysis"]).exists()
    assert (dirs["logs"]).exists()

