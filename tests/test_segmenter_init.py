from adapt.radar.cell_segmenter import RadarCellSegmenter
import pytest


def test_default_config():
    seg = RadarCellSegmenter({})
    assert seg.method == "threshold"
    assert seg.threshold == 30
    assert seg.filter_by_size is True


def test_custom_config():
    seg = RadarCellSegmenter({
        "method": "threshold",
        "threshold": 45,
        "min_cellsize_gridpoint": 10,
        "filter_by_size": False,
    })

    assert seg.threshold == 45
    assert seg.min_gridpoints == 10
    assert seg.filter_by_size is False


def test_unknown_method_raises():
    seg = RadarCellSegmenter({"method": "watershed"})
    with pytest.raises(ValueError):
        seg.segment(None)
