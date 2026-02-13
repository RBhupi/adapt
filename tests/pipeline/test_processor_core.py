
import numpy as np
import xarray as xr
from adapt.pipeline.processor import RadarProcessor


import pytest

pytestmark = [pytest.mark.unit, pytest.mark.pipeline]


def fake_ds():
    data = np.zeros((1, 1, 10, 10))
    return xr.Dataset(
        {
            "reflectivity": (("time", "z", "y", "x"), data)
        },
        coords={
            "time": [0],
            "z": [2000],
            "y": range(10),
            "x": range(10),
        }
    )


def test_extract_2d_slice(processor_queues, pipeline_config, pipeline_output_dirs):
    in_q, _ = processor_queues
    proc = RadarProcessor(in_q, pipeline_config, pipeline_output_dirs)

    ds = fake_ds()
    ds2d = proc._extract_2d_slice(ds)

    assert "reflectivity" in ds2d
    assert ds2d["reflectivity"].ndim == 2
    assert ds2d.attrs["z_level_m"] == 2000


def test_compute_projections_first_frame_noop(processor_queues, pipeline_config, pipeline_output_dirs):
    in_q, _ = processor_queues
    proc = RadarProcessor(in_q, pipeline_config, pipeline_output_dirs)

    ds = fake_ds()
    ds2d = proc._extract_2d_slice(ds)

    out = proc._compute_projections(ds2d, "file1")
    assert out is ds2d  # no projections on first frame
