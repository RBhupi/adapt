import numpy as np
from adapt.radar.cell_projector import RadarCellProjector


import pytest

pytestmark = pytest.mark.unit


def test_projection_adds_expected_variables(simple_labeled_ds_pair, make_config):
    """Projector adds expected projection variables."""
    from adapt.schemas.user import UserProjectorConfig
    config = make_config(projector=UserProjectorConfig(max_projection_steps=1))
    proj = RadarCellProjector(config)

    out = proj.project(simple_labeled_ds_pair)

    assert "cell_projections" in out
    assert "heading_x" in out
    assert "heading_y" in out


def test_projection_dimensions(simple_labeled_ds_pair, make_config):
    """Projection output has correct dimensions."""
    from adapt.schemas.user import UserProjectorConfig
    config = make_config(projector=UserProjectorConfig(max_projection_steps=2))
    proj = RadarCellProjector(config)

    out = proj.project(simple_labeled_ds_pair)

    proj_da = out["cell_projections"]

    assert proj_da.dims == ("frame_offset", "y", "x")
    assert proj_da.shape[0] == 3  # 1 registration + 2 future
