import pytest

pytestmark = pytest.mark.unit
from adapt.radar.cell_analyzer import RadarCellAnalyzer


def test_extract_requires_cell_labels(simple_2d_ds, make_config):
    """Analyzer raises error when cell_labels variable is missing."""
    config = make_config()
    analyzer = RadarCellAnalyzer(config)

    with pytest.raises(ValueError, match="cell_labels"):
        analyzer.extract(simple_2d_ds)

