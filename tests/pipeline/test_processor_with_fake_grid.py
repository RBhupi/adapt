import queue
from pathlib import Path

from adapt.pipeline.processor import RadarProcessor
from adapt.setup_directories import setup_output_directories
import pytest
from tests.helpers.fake_grid import make_fake_grid_ds_with_labels

pytestmark = [pytest.mark.unit, pytest.mark.pipeline]


def test_processor_accepts_fake_grid(tmp_path, monkeypatch, pipeline_config, pipeline_output_dirs):
    """Processor can process fake grid datasets for testing."""
    in_q = queue.Queue()

    proc = RadarProcessor(in_q, pipeline_config, pipeline_output_dirs)

    fake_grid = make_fake_grid_ds_with_labels()

    # ---- Patch filesystem ----
    monkeypatch.setattr(
        "adapt.pipeline.processor.Path.exists",
        lambda self: True
    )

    # ---- Patch loader boundary (correct seam) ----
    monkeypatch.setattr(
        proc,
        "_load_and_regrid",
        lambda filepath: (
            fake_grid,          # ds
            fake_grid,          # ds_2d
            tmp_path / "fake.nc",  # nc_full_path
            None,               # scan_time
        )
    )

    # ---- Patch persistence boundary ----
    monkeypatch.setattr(
        proc,
        "_analyze_and_save",
        lambda *a, **k: True
    )

    ok = proc.process_file("/fake/file")

    assert ok is True
