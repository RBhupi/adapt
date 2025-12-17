import pytest
from adapt.radar.cell_analyzer import RadarCellAnalyzer

pytestmark = pytest.mark.unit


def test_init_with_default_config(make_config):
    """Analyzer initializes with default config."""
    config = make_config()
    analyzer = RadarCellAnalyzer(config)

    assert analyzer.reflectivity_field == "reflectivity"
    assert analyzer.max_projection_steps > 0


def test_init_custom_config(make_config):
    """Analyzer initializes with custom config."""
    from adapt.schemas.user import UserProjectorConfig, UserGlobalConfig
    config = make_config(
        reflectivity_var="dbz",
        projector=UserProjectorConfig(max_projection_steps=2)
    )
    analyzer = RadarCellAnalyzer(config)

    assert analyzer.reflectivity_field == "dbz"
    assert analyzer.max_projection_steps == 2