import pytest
from adapt.radar.cell_projector import RadarCellProjector

pytestmark = pytest.mark.unit

def test_max_projection_steps_is_capped(simple_labeled_ds_pair, make_config):
    """Projection steps are capped at default maximum."""
    from adapt.schemas.user import UserProjectorConfig
    config = make_config(projector=UserProjectorConfig(max_projection_steps=100))
    proj = RadarCellProjector(config)

    out = proj.project(simple_labeled_ds_pair)

    assert out["cell_projections"].shape[0] == 11  # 1 + 10


def test_custom_flow_params_do_not_crash(simple_labeled_ds_pair, make_config):
    """Custom flow parameters are accepted without crashing."""
    from adapt.schemas.user import UserProjectorConfig
    config = make_config(
        projector=UserProjectorConfig(
            flow_params={
                "pyr_scale": 0.3,
                "levels": 2,
                "winsize": 5,
                "iterations": 2,
                "poly_n": 3,
                "poly_sigma": 1.1,
                "flags": 0,
            }
        )
    )
    proj = RadarCellProjector(config)

    out = proj.project(simple_labeled_ds_pair)

    assert "cell_projections" in out
