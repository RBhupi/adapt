from adapt.radar.cell_segmenter import RadarCellSegmenter


def test_close_cells_without_closing(close_cells_ds):
    seg = RadarCellSegmenter({
        "threshold": 30,
        "filter_by_size": False,
    })

    labels = seg.segment(close_cells_ds)["cell_labels"].values

    assert labels.max() == 2


def test_close_cells_with_closing(close_cells_ds):
    seg = RadarCellSegmenter({
        "threshold": 30,
        "closing_kernel": (2, 2),
        "filter_by_size": False,
    })

    labels = seg.segment(close_cells_ds)["cell_labels"].values

    # Expected to merge into one object
    assert labels.max() == 1

