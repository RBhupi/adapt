"""Extended tests for RadarDataLoader functionality."""

import pytest
from pathlib import Path
from adapt.radar.loader import RadarDataLoader

pytestmark = pytest.mark.unit


def test_loader_stores_config(make_config):
    """Test that loader stores config correctly."""
    config = make_config()
    loader = RadarDataLoader(config)
    
    assert loader.config is not None
    assert hasattr(loader.config, 'regridder')
    assert hasattr(loader.config, 'downloader')


def test_loader_with_custom_grid_shape(make_config):
    """Test loader with custom grid configuration."""
    config = make_config(grid_shape=(10, 50, 50))
    loader = RadarDataLoader(config)
    
    assert loader.config.regridder.grid_shape == (10, 50, 50)


def test_loader_with_custom_z_level(make_config):
    """Test loader uses configured z_level (global)."""
    config = make_config(z_level=3000.0)
    loader = RadarDataLoader(config)

    assert loader.config.global_.z_level == 3000.0

def test_loader_with_custom_weighting_function(make_config):
    """Test loader respects weighting function config (via nested regridder key)."""
    config = make_config(regridder={"weighting_function": "barnes"})
    loader = RadarDataLoader(config)

    assert loader.config.regridder.weighting_function == "barnes"


def test_loader_with_custom_min_radius(make_config):
    """Test loader respects min_radius config (via nested regridder key)."""
    config = make_config(regridder={"min_radius": 2000.0})
    loader = RadarDataLoader(config)

    assert loader.config.regridder.min_radius == 2000.0


def test_loader_with_custom_roi_func(make_config):
    """Test loader respects roi_func config (via nested regridder key)."""
    # 'dist' is a valid roi_func alternative to default 'dist_beam'
    config = make_config(regridder={"roi_func": "dist"})
    loader = RadarDataLoader(config)

    assert loader.config.regridder.roi_func == "dist"




def test_loader_with_custom_grid_limits(make_config):
    """Test loader with custom grid limits."""
    config = make_config(
        grid_limits=((0, 10000), (-50000, 50000), (-50000, 50000))
    )
    loader = RadarDataLoader(config)
    
    assert loader.config.regridder.grid_limits[0] == (0, 10000)


def test_loader_initialization_succeeds(make_config):
    """Test that loader can be created successfully."""
    config = make_config()
    loader = RadarDataLoader(config)
    
    assert loader is not None


def test_read_nonexistent_file_returns_none(tmp_path, make_config):
    """Test reading non-existent file returns None."""
    config = make_config()
    loader = RadarDataLoader(config)
    
    result = loader.read(tmp_path / "nonexistent.nc")
    assert result is None
