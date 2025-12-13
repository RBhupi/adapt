import pytest
from adapt.radar.loader import RadarDataLoader

def test_loader_requires_config():
    with pytest.raises(ValueError):
        RadarDataLoader(None)

def test_loader_requires_reader_and_regridder():
    with pytest.raises(KeyError):
        RadarDataLoader({"reader": {}})


def test_read_missing_file_returns_none(basic_config):
    loader = RadarDataLoader(basic_config)
    assert loader.read("/does/not/exist") is None

def test_regrid_handles_exception(monkeypatch, basic_config):
    loader = RadarDataLoader(basic_config)

    def boom(*a, **k):
        raise RuntimeError("fail")

    monkeypatch.setattr("pyart.map.grid_from_radars", boom)

    out = loader.regrid(object())
    assert out is None
