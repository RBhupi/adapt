import pytest

pytestmark = pytest.mark.unit


def test_heading_statistics_optional(labeled_ds_with_extras, make_config):
    """Heading statistics are included in extraction."""
    from adapt.radar.cell_analyzer import RadarCellAnalyzer

    config = make_config()
    analyzer = RadarCellAnalyzer(config)
    df = analyzer.extract(labeled_ds_with_extras)

    assert "cell_heading_x_mean" in df.columns
    assert "cell_heading_y_mean" in df.columns


def test_projection_centroids_json_present(labeled_ds_with_extras, make_config):
    """Projection centroids are included in extraction."""
    from adapt.radar.cell_analyzer import RadarCellAnalyzer

    config = make_config()
    analyzer = RadarCellAnalyzer(config)

    df = analyzer.extract(labeled_ds_with_extras)

    assert "cell_projection_centroids_json" in df.columns
    assert isinstance(df.iloc[0]["cell_projection_centroids_json"], str)
