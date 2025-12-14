import pytest
from adapt.radar.cell_projector import RadarCellProjector

pytestmark = pytest.mark.unit

def test_max_projection_steps_is_capped(simple_labeled_ds_pair):
    proj = RadarCellProjector({
        "method": "adapt_default",
        "max_projection_steps": 100,  # capped at 10
    })

    out = proj.project(simple_labeled_ds_pair)

    assert out["cell_projections"].shape[0] == 11  # 1 + 10


def test_custom_flow_params_do_not_crash(simple_labeled_ds_pair):
    proj = RadarCellProjector({
        "method": "adapt_default",
        "flow_params": {
            "pyr_scale": 0.3,
            "levels": 2,
            "winsize": 5,
            "iterations": 2,
            "poly_n": 3,
            "poly_sigma": 1.1,
            "flags": 0,
        }
    })

    out = proj.project(simple_labeled_ds_pair)

    assert "cell_projections" in out
