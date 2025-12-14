import pytest
import numpy as np

pytestmark = pytest.mark.unit
from adapt.radar.cell_analyzer import RadarCellAnalyzer


def test_get_lat_lon_bounds():
    lat = np.ones((5, 5))
    lon = np.ones((5, 5))

    lat_val, lon_val = RadarCellAnalyzer.get_lat_lon(100, 100, lat, lon)

    assert np.isnan(lat_val)
    assert np.isnan(lon_val)


def test_pixel_area_computation(simple_2d_ds):
    analyzer = RadarCellAnalyzer()

    area = analyzer._pixel_area_km2(simple_2d_ds)

    assert area > 0
