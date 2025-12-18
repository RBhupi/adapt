"""Test RadarCellSegmenter output contract and data structure."""

import pytest
import numpy as np
from adapt.radar.cell_segmenter import RadarCellSegmenter

pytestmark = pytest.mark.unit


def test_output_contract(simple_2d_ds, internal_config):
    """Segmenter output has correct shape, dtype, and metadata."""
    seg = RadarCellSegmenter(internal_config)

    out = seg.segment(simple_2d_ds)
    da = out["cell_labels"]

    assert da.dims == ("y", "x")
    assert da.dtype == np.int32
    assert "threshold" in da.attrs
    assert "z_level_m" in da.attrs
    assert da.attrs["method"] == "threshold"
