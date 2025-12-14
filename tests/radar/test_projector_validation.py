import numpy as np
import pytest
from adapt.radar.cell_projector import RadarCellProjector


import pytest

pytestmark = pytest.mark.unit


def test_validate_requires_two_datasets(simple_labeled_ds_pair):
    proj = RadarCellProjector({"method": "adapt_default"})

    with pytest.raises(ValueError):
        proj.project(simple_labeled_ds_pair[:1])


def test_projection_skipped_if_time_gap_too_large(simple_labeled_ds_pair):
    proj = RadarCellProjector({
        "method": "adapt_default",
        "max_time_interval_minutes": 1,  # too strict
    })

    out = proj.project(simple_labeled_ds_pair)

    # No projections added
    assert "cell_projections" not in out

