"""Test RadarCellSegmenter morphological operations."""

from adapt.radar.cell_segmenter import RadarCellSegmenter
import pytest

pytestmark = pytest.mark.unit


def test_close_cells_without_closing(close_cells_ds, make_config):
    """Without morphological closing, nearby cells remain separate."""
    from adapt.schemas.user import UserSegmenterConfig
    config = make_config(
        threshold_dbz=30, 
        segmenter=UserSegmenterConfig(filter_by_size=False)
    )
    seg = RadarCellSegmenter(config)

    labels = seg.segment(close_cells_ds)["cell_labels"].values

    # Two cells separated by gap should remain separate
    assert labels.max() == 2


def test_close_cells_with_closing(close_cells_ds, make_config):
    """Morphological closing (2,2) merges nearby cells across 1-pixel gap."""
    # Note: closing_kernel not yet exposed in UserConfig, need to use internal access
    from adapt.schemas.user import UserSegmenterConfig
    config = make_config(
        threshold_dbz=30, 
        segmenter=UserSegmenterConfig(filter_by_size=False, closing_kernel=(2, 2))
    )
    seg = RadarCellSegmenter(config)

    labels = seg.segment(close_cells_ds)["cell_labels"].values

    # Closing (2,2) should merge the two cells into one
    assert labels.max() == 1


