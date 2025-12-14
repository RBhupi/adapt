def test_heading_statistics_optional(labeled_ds_with_extras):
    from adapt.radar.cell_analyzer import RadarCellAnalyzer

    analyzer = RadarCellAnalyzer()
    df = analyzer.extract(labeled_ds_with_extras)

    assert "cell_heading_x_mean" in df.columns
    assert "cell_heading_y_mean" in df.columns


def test_projection_centroids_json_present(labeled_ds_with_extras):
    from adapt.radar.cell_analyzer import RadarCellAnalyzer

    analyzer = RadarCellAnalyzer({
        "projector": {"max_projection_steps": 1}
    })

    df = analyzer.extract(labeled_ds_with_extras)

    assert "cell_projection_centroids_json" in df.columns
    assert isinstance(df.iloc[0]["cell_projection_centroids_json"], str)
