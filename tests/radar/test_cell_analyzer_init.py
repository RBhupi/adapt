import pytest
from adapt.radar.cell_analyzer import RadarCellAnalyzer

pytestmark = pytest.mark.unit


def test_init_with_default_config():
    analyzer = RadarCellAnalyzer()

    assert analyzer.reflectivity_field == "reflectivity"
    assert analyzer.max_projection_steps > 0


def test_init_custom_config():
    analyzer = RadarCellAnalyzer({
        "global": {
            "var_names": {
                "reflectivity": "dbz"
            }
        },
        "projector": {
            "max_projection_steps": 2
        }
    })

    assert analyzer.reflectivity_field == "dbz"
    assert analyzer.max_projection_steps == 2