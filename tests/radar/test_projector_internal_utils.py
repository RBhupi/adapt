import numpy as np
from adapt.radar.cell_projector import RadarCellProjector


import pytest

pytestmark = pytest.mark.unit


def test_normalize_constant_field():
    proj = RadarCellProjector({})

    a = np.ones((4, 4), dtype=np.float32) * 10
    b = np.ones((4, 4), dtype=np.float32) * 10

    a_n, b_n = proj._normalize(a, b)

    assert a_n.dtype == np.uint8
    assert b_n.dtype == np.uint8


def test_fill_concave_hull_small_object_falls_back():
    proj = RadarCellProjector({})

    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True

    filled = proj._fill_concave_hull(mask)

    assert filled.any()
