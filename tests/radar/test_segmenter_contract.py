import pytest
from adapt.radar.cell_segmenter import RadarCellSegmenter

pytestmark = pytest.mark.unit
import numpy as np


def test_output_contract(simple_2d_ds):
    seg = RadarCellSegmenter({"threshold": 30})

    out = seg.segment(simple_2d_ds)
    da = out["cell_labels"]

    assert da.dims == ("y", "x")
    assert da.dtype == np.int32
    assert "threshold_dbz" in da.attrs
    assert "z_level_m" in da.attrs
    assert da.attrs["method"] == "threshold"
