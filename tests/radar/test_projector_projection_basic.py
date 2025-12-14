import numpy as np
from adapt.radar.cell_projector import RadarCellProjector


def test_projection_adds_expected_variables(simple_labeled_ds_pair):
    proj = RadarCellProjector({
        "method": "adapt_default",
        "max_projection_steps": 1,
    })

    out = proj.project(simple_labeled_ds_pair)

    assert "cell_projections" in out
    assert "heading_x" in out
    assert "heading_y" in out


def test_projection_dimensions(simple_labeled_ds_pair):
    proj = RadarCellProjector({
        "method": "adapt_default",
        "max_projection_steps": 2,
    })

    out = proj.project(simple_labeled_ds_pair)

    proj_da = out["cell_projections"]

    assert proj_da.dims == ("frame_offset", "y", "x")
    assert proj_da.shape[0] == 3  # 1 registration + 2 future
