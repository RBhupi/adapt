"""Test RadarCellSegmenter initialization with Pydantic configs."""

from adapt.radar.cell_segmenter import RadarCellSegmenter
from adapt.schemas import ParamConfig, UserConfig, resolve_config
import pytest


pytestmark = pytest.mark.unit


def test_default_config(internal_config):
    """Segmenter uses expert defaults when no user overrides provided."""
    seg = RadarCellSegmenter(internal_config)
    assert seg.method == "threshold"
    assert seg.threshold == 30.0
    assert seg.filter_by_size is True


def test_custom_config(make_config):
    """Segmenter respects user config overrides."""
    config = make_config(
        threshold_dbz=45,
        min_cell_size=10,
        # Note: filter_by_size not exposed in UserConfig yet, uses default
    )
    
    seg = RadarCellSegmenter(config)
    assert seg.threshold == 45.0
    assert seg.min_gridpoints == 10


def test_unknown_method_raises():
    """Invalid segmentation method fails at config validation time."""
    # Pydantic validation happens at model creation, not at runtime
    # This test verifies the old behavior is no longer needed
    # Invalid methods are caught by Literal["threshold"] in ParamConfig
    
    with pytest.raises(Exception):  # ValidationError from Pydantic
        # Try to create config with invalid method
        param = ParamConfig()
        user = UserConfig(segmentation_method="watershed")  # Invalid
        resolve_config(param, user, None)

