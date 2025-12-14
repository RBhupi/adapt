import pytest
from adapt.radar.cell_segmenter import RadarCellSegmenter

pytestmark = pytest.mark.unit
import pytest
import xarray as xr
import numpy as np


def test_missing_reflectivity_var():
    ds = xr.Dataset(
        {"wrong_var": (("y", "x"), np.ones((3, 3)))}
    )

    seg = RadarCellSegmenter({})
    with pytest.raises(KeyError):
        seg.segment(ds)


def test_non_2d_data_fails():
    ds = xr.Dataset(
        {"reflectivity": (("z", "y", "x"), np.ones((2, 3, 3)))}
    )

    seg = RadarCellSegmenter({})
    with pytest.raises(Exception):
        seg.segment(ds)
