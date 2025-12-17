"""Test RadarCellSegmenter error handling and edge cases."""

import pytest
import xarray as xr
import numpy as np
from adapt.radar.cell_segmenter import RadarCellSegmenter

pytestmark = pytest.mark.unit


def test_missing_reflectivity_var(internal_config):
    """Segmenter fails gracefully when reflectivity variable missing."""
    ds = xr.Dataset(
        {"wrong_var": (("y", "x"), np.ones((3, 3)))}
    )

    seg = RadarCellSegmenter(internal_config)
    with pytest.raises(KeyError):
        seg.segment(ds)


def test_non_2d_data_fails(internal_config):
    """Segmenter rejects 3D data (must be 2D slice)."""
    ds = xr.Dataset(
        {"reflectivity": (("z", "y", "x"), np.ones((2, 3, 3)))}
    )

    seg = RadarCellSegmenter(internal_config)
    with pytest.raises(Exception):
        seg.segment(ds)

