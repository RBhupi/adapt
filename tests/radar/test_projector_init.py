import pytest
import pytest
from adapt.radar.cell_projector import RadarCellProjector

pytestmark = pytest.mark.unit


def test_init_stores_config():
    cfg = {"method": "adapt_default"}
    proj = RadarCellProjector(cfg)

    assert proj.config is cfg


def test_unknown_method_raises():
    proj = RadarCellProjector({"method": "unknown"})

    with pytest.raises(ValueError, match="Unknown projection method"):
        proj.project([])
