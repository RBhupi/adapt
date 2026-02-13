import pytest
import pytest
from adapt.radar.cell_projector import RadarCellProjector

pytestmark = pytest.mark.unit


def test_init_stores_config(make_config):
    """Projector stores config reference."""
    config = make_config()
    proj = RadarCellProjector(config)

    assert proj.config is config
