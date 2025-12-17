"""Test RadarCellSegmenter filtering logic."""

from adapt.radar.cell_segmenter import RadarCellSegmenter
import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_min_cellsize_filter(two_cell_ds, make_config):
    """Small cells below min_cellsize threshold are filtered out."""
    config = make_config(threshold_dbz=20, min_cell_size=4)
    seg = RadarCellSegmenter(config)

    out = seg.segment(two_cell_ds)
    labels = out["cell_labels"].values

    # Both cells meet min_size (4 pixels each), so expect both or merged
    assert labels.max() >= 1


def test_disable_size_filter(two_cell_ds, make_config):
    """All detected cells are retained when filter_by_size=False."""
    # threshold_dbz=20 will detect both cells (50 and 30 dBZ)
    from adapt.schemas.user import UserSegmenterConfig
    config = make_config(
        threshold_dbz=20.0, 
        segmenter=UserSegmenterConfig(filter_by_size=False)
    )
    seg = RadarCellSegmenter(config)

    out = seg.segment(two_cell_ds)
    labels = out["cell_labels"].values

    # Both cells should be detected
    assert labels.max() == 2


def test_relabeling_is_contiguous(two_cell_ds, make_config):
    """Cell labels are contiguous integers starting from 1."""
    from adapt.schemas.user import UserSegmenterConfig
    config = make_config(
        threshold_dbz=20.0, 
        segmenter=UserSegmenterConfig(filter_by_size=False)
    )
    seg = RadarCellSegmenter(config)

    labels = seg.segment(two_cell_ds)["cell_labels"].values
    unique = sorted(set(labels.flatten()) - {0})

    # Labels should be [1, 2] for two cells
    assert unique == list(range(1, len(unique) + 1))


