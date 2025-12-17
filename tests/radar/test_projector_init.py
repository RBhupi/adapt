import pytest
import pytest
from adapt.radar.cell_projector import RadarCellProjector

pytestmark = pytest.mark.unit


def test_init_stores_config(make_config):
    """Projector stores config reference."""
    config = make_config()
    proj = RadarCellProjector(config)

    assert proj.config is config


def test_unknown_method_raises(make_config):
    """Projector raises on unknown projection method."""
    from adapt.schemas.user import UserProjectorConfig
    config = make_config(projector=UserProjectorConfig(method="unknown"))
    proj = RadarCellProjector(config)

    with pytest.raises(ValueError, match="Unknown projection method"):
        proj.project([])
