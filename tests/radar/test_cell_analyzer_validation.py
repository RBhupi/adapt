import pytest

pytestmark = pytest.mark.unit
from adapt.radar.cell_analyzer import RadarCellAnalyzer


def test_extract_requires_cell_labels(labeled_ds_with_extras, make_config):
    """Analyzer works correctly when cell_labels variable is present.
    
    NOTE: This replaces the old defensive check test. After SRP refactoring,
    the analyzer no longer validates input - it assumes the segmenter has
    already added labels. Input validation is Pydantic's responsibility.
    """
    config = make_config()
    analyzer = RadarCellAnalyzer(config)

    # With labels present, extract should work
    df = analyzer.extract(labeled_ds_with_extras)
