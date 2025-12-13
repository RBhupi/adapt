from adapt.radar.cell_segmenter import RadarCellSegmenter
import numpy as np


def test_min_cellsize_filter(two_cell_ds):
    seg = RadarCellSegmenter({
        "threshold": 20,
        "min_cellsize_gridpoint": 4,
    })

    out = seg.segment(two_cell_ds)
    labels = out["cell_labels"].values

    # only the larger cell should remain
    assert labels.max() == 2


def test_disable_size_filter(two_cell_ds):
    seg = RadarCellSegmenter({
        "threshold": 20,
        "filter_by_size": False,
    })

    out = seg.segment(two_cell_ds)
    labels = out["cell_labels"].values

    assert labels.max() == 2


def test_relabeling_is_contiguous(two_cell_ds):
    seg = RadarCellSegmenter({
        "threshold": 20,
        "filter_by_size": False,
    })

    labels = seg.segment(two_cell_ds)["cell_labels"].values
    unique = sorted(set(labels.flatten()) - {0})

    assert unique == [1, 2]
