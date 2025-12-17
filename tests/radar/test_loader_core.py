import pytest

pytestmark = pytest.mark.unit
from adapt.radar.loader import RadarDataLoader


# Note: Legacy tests for None/incomplete dict configs removed.
# InternalConfig validation now prevents invalid configurations at creation time.


def test_read_missing_file_returns_none(radar_config):
    """Loader returns None for missing files."""
    loader = RadarDataLoader(radar_config)
    assert loader.read("/does/not/exist") is None


def test_regrid_handles_exception(monkeypatch, radar_config):
    """Loader handles regridding exceptions gracefully."""
    loader = RadarDataLoader(radar_config)

    def boom(*a, **k):
        raise RuntimeError("fail")

    monkeypatch.setattr("pyart.map.grid_from_radars", boom)

    out = loader.regrid(object())
    assert out is None
