from adapt.radar.cell_segmenter import RadarCellSegmenter
import numpy as np



def test_threshold_filters_all(simple_2d_ds):
    seg = RadarCellSegmenter({
        "threshold": 30,
        "closing_kernel": (1, 1),  # explicit
    })

    out = seg.segment(simple_2d_ds)

    assert "cell_labels" in out

    labels = out["cell_labels"].values

    # At least one cell should exist
    assert labels.max() == 0
    assert np.count_nonzero(labels) == 0



def test_threshold_creates_at_least_one_cell(simple_2d_ds):
    seg = RadarCellSegmenter({
        "threshold": 30,
        "closing_kernel": (1, 1),
        "min_cellsize_gridpoint": 2,
    })

    out = seg.segment(simple_2d_ds)

    assert "cell_labels" in out

    labels = out["cell_labels"].values

    assert labels.max() >= 1
    assert np.count_nonzero(labels) > 0


def test_no_cells_below_threshold(empty_2d_ds):
    seg = RadarCellSegmenter({"threshold": 30})

    out = seg.segment(empty_2d_ds)
    labels = out["cell_labels"].values

    assert labels.max() == 0


def test__multiple_cells(large_multi_cell_ds):
    seg = RadarCellSegmenter({
        "threshold": 30,
        "filter_by_size": False,
    })

    out = seg.segment(large_multi_cell_ds)
    labels = out["cell_labels"].values

    # Expect four distinct cells
    assert labels.max() == 4