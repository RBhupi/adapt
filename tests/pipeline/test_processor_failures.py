from adapt.pipeline.processor import RadarProcessor
from pathlib import Path
import queue
import pytest

def test_process_missing_file(pipeline_config, pipeline_output_dirs):
    """Processor handles missing file gracefully."""
    q = queue.Queue()
    proc = RadarProcessor(q, pipeline_config, pipeline_output_dirs)

    ok = proc.process_file("/does/not/exist")
    assert ok is False


def test_loader_returns_none(monkeypatch, processor_queues, pipeline_config, pipeline_output_dirs):
    """Processor handles None return from loader."""
    in_q, _ = processor_queues
    proc = RadarProcessor(in_q, pipeline_config, pipeline_output_dirs)

    monkeypatch.setattr(
        proc.loader,
        "load_and_regrid",
        lambda *a, **k: None
    )

    ok = proc.process_file("/fake/path/file")
    assert ok is False


def test_processor_handles_loader_exception(monkeypatch, processor_queues, pipeline_config, pipeline_output_dirs):
    """Processor handles loader exceptions gracefully."""
    in_q, _ = processor_queues
    proc = RadarProcessor(in_q, pipeline_config, pipeline_output_dirs)

    def boom(*a, **k):
        raise IOError("disk failure")

    monkeypatch.setattr(proc.loader, "load_and_regrid", boom)

    ok = proc.process_file("/fake/path/file")

    assert ok is False

# following test (test_processor_enqueues_when_netcdf_is_written) was most painful to write and pass .
def make_fake_grid_ds(with_labels=True):
    data = {
        "reflectivity": (("y", "x"), np.ones((4, 4)))
    }

    if with_labels:
        data["cell_labels"] = (("y", "x"), np.array([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
        ]))

    ds = xr.Dataset(
        data,
        coords={
            "x": np.arange(4),
            "y": np.arange(4),
        },
        attrs={"z_level_m": 2000}
    )
    return ds

import numpy as np
import xarray as xr
def test_processor_enqueues_when_netcdf_is_written(
    tmp_path,
    monkeypatch,
    processor_queues,
    pipeline_config,
    pipeline_output_dirs,
):
    """Processor saves NetCDF and returns success.

    Note: With the new architecture, processor no longer enqueues to plotter.
    PlotConsumer polls the repository independently.
    """
    in_q, _ = processor_queues
    proc = RadarProcessor(in_q, pipeline_config, pipeline_output_dirs)

    # ---- fake filesystem ----
    monkeypatch.setattr(
        "adapt.pipeline.processor.Path.exists",
        lambda self: True
    )

    # ---- fake dataset ----
    fake_ds = make_fake_grid_ds(with_labels=True)

    # ---- patch loader ----
    monkeypatch.setattr(
        proc,
        "_load_and_regrid",
        lambda filepath: (
            fake_ds,
            fake_ds,
            tmp_path / "grid.nc",
            None,
        )
    )

    # ---- identity segmentation ----
    monkeypatch.setattr(proc.segmenter, "segment", lambda ds: ds)

    # ---- no-op projections ----
    monkeypatch.setattr(proc, "_compute_projections", lambda ds, _: ds)

    # ---- real NetCDF save ----
    seg_path = tmp_path / "seg.nc"

    import pytest
    from tests.helpers.fake_netcdf import write_fake_segmentation_netcdf

    pytestmark = [pytest.mark.unit, pytest.mark.pipeline]

    monkeypatch.setattr(
        proc,
        "_save_segmentation_netcdf",
        lambda *a, **k: str(
            write_fake_segmentation_netcdf(seg_path)
        )
    )

    # ---- analysis succeeds ----
    monkeypatch.setattr(proc, "_analyze_and_save", lambda *a, **k: True)

    ok = proc.process_file("/fake/file")

    assert ok is True
