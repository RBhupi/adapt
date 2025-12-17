"""Test RadarCellSegmenter threshold-based segmentation."""

import pytest
import numpy as np
from adapt.radar.cell_segmenter import RadarCellSegmenter

pytestmark = pytest.mark.unit


def test_threshold_filters_all(simple_2d_ds, make_config):
    """Threshold higher than max value results in no cells."""
    config = make_config(threshold_dbz=50)  # Higher than 40 in simple_2d_ds
    seg = RadarCellSegmenter(config)

    out = seg.segment(simple_2d_ds)

    assert "cell_labels" in out
    labels = out["cell_labels"].values

    # No cells should exist
    assert labels.max() == 0
    assert np.count_nonzero(labels) == 0


def test_threshold_creates_at_least_one_cell(simple_2d_ds, make_config):
    """Threshold below max value creates cells."""
    config = make_config(threshold_dbz=30, min_cell_size=2)
    seg = RadarCellSegmenter(config)

    out = seg.segment(simple_2d_ds)

    assert "cell_labels" in out
    labels = out["cell_labels"].values

    assert labels.max() >= 1
    assert np.count_nonzero(labels) > 0


def test_no_cells_below_threshold(empty_2d_ds, internal_config):
    """Empty dataset (all zeros) produces no cells."""
    seg = RadarCellSegmenter(internal_config)

    out = seg.segment(empty_2d_ds)
    labels = out["cell_labels"].values

    assert labels.max() == 0


def test__multiple_cells(large_multi_cell_ds, make_config):
    """Multiple distinct cells are detected and labeled."""
    # Don't filter by size for this test
    from adapt.schemas.user import UserSegmenterConfig
    config = make_config(
        threshold_dbz=30, 
        segmenter=UserSegmenterConfig(filter_by_size=False)
    )
    seg = RadarCellSegmenter(config)

    out = seg.segment(large_multi_cell_ds)
    labels = out["cell_labels"].values

    # Expect four distinct cells
    assert labels.max() == 4
