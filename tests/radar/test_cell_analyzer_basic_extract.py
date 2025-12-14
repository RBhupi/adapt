import pytest
import pandas as pd

pytestmark = pytest.mark.unit
from adapt.radar.cell_analyzer import RadarCellAnalyzer


def test_extract_single_cell(labeled_ds_with_extras):
    analyzer = RadarCellAnalyzer()

    df = analyzer.extract(labeled_ds_with_extras)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["cell_label"] == 1


def test_extract_produces_required_columns(labeled_ds_with_extras):
    analyzer = RadarCellAnalyzer()

    df = analyzer.extract(labeled_ds_with_extras)
    row = df.iloc[0]

    assert "cell_area_sqkm" in row
    assert "cell_centroid_geom_x" in row
    assert "cell_centroid_geom_y" in row
