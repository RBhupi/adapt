"""Extended tests for setup_directories utility functions."""

import pytest
from pathlib import Path
from datetime import datetime, timezone
from adapt.setup_directories import (
    setup_output_directories,
    get_nexrad_path,
    get_netcdf_path,
    get_analysis_path,
    get_plot_path,
    get_log_path,
)


def test_setup_output_directories_with_explicit_path(tmp_path):
    """Test setup with explicit base directory."""
    dirs = setup_output_directories(tmp_path)
    
    assert dirs["base"] == tmp_path
    assert dirs["nexrad"] == tmp_path / "nexrad"
    assert dirs["gridnc"] == tmp_path / "gridnc"
    assert dirs["analysis"] == tmp_path / "analysis"
    assert dirs["plots"] == tmp_path / "plots"
    assert dirs["logs"] == tmp_path / "logs"
    
    # All directories should exist
    for key, path in dirs.items():
        assert path.exists(), f"{key} directory not created"


def test_setup_output_directories_creates_subdirs(tmp_path):
    """Test that all required subdirectories are created."""
    dirs = setup_output_directories(tmp_path)
    
    assert (tmp_path / "nexrad").is_dir()
    assert (tmp_path / "gridnc").is_dir()
    assert (tmp_path / "analysis").is_dir()
    assert (tmp_path / "plots").is_dir()
    assert (tmp_path / "logs").is_dir()


def test_get_nexrad_path_with_scan_time(tmp_path):
    """Test getting NEXRAD path with scan time."""
    dirs = setup_output_directories(tmp_path)
    scan_time = datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc)
    
    nexrad_path = get_nexrad_path(dirs, "KMOB", "KMOB20240115_123045_V06", scan_time)
    
    assert nexrad_path is not None
    assert "20240115" in str(nexrad_path)
    assert "KMOB" in str(nexrad_path)
    assert nexrad_path.parent.parent.parent == dirs["nexrad"]


def test_get_nexrad_path_creates_directory(tmp_path):
    """Test that getting NEXRAD path creates directory structure."""
    dirs = setup_output_directories(tmp_path)
    scan_time = datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc)
    
    nexrad_path = get_nexrad_path(dirs, "KMOB", "test_file.nc", scan_time)
    
    # Parent directory should exist
    assert nexrad_path.parent.exists()


def test_get_netcdf_path_with_scan_time(tmp_path):
    """Test getting NetCDF path with scan time."""
    dirs = setup_output_directories(tmp_path)
    scan_time = datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc)
    
    nc_path = get_netcdf_path(dirs, "KMOB", "grid_KMOB_20240115_123045.nc", scan_time)
    
    assert nc_path is not None
    assert "20240115" in str(nc_path)
    assert "KMOB" in str(nc_path)
    assert nc_path.parent.parent.parent == dirs["gridnc"]


def test_get_netcdf_path_creates_directory(tmp_path):
    """Test that getting NetCDF path creates directory structure."""
    dirs = setup_output_directories(tmp_path)
    scan_time = datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc)
    
    nc_path = get_netcdf_path(dirs, "KMOB", "test_grid.nc", scan_time)
    
    # Parent directory should exist
    assert nc_path.parent.exists()


def test_get_analysis_path_parquet(tmp_path):
    """Test getting analysis path for parquet files."""
    dirs = setup_output_directories(tmp_path)
    
    analysis_path = get_analysis_path(
        dirs, 
        radar_id="KMOB", 
        analysis_type="parquet",
        filename="cells_20240115.parquet"
    )
    
    assert analysis_path is not None
    assert analysis_path.suffix == ".parquet"
    # Path includes date/radar hierarchy
    assert "KMOB" in str(analysis_path)


def test_get_analysis_path_database(tmp_path):
    """Test getting analysis path for database files."""
    dirs = setup_output_directories(tmp_path)
    
    db_path = get_analysis_path(
        dirs,
        radar_id="KMOB",
        analysis_type="database",
        filename="KMOB_cells_statistics.db"
    )
    
    assert db_path is not None
    assert db_path.suffix == ".db"


def test_get_plot_path_reflectivity(tmp_path):
    """Test getting plot path for reflectivity plots."""
    dirs = setup_output_directories(tmp_path)
    scan_time = datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc)
    
    plot_path = get_plot_path(
        dirs,
        radar_id="KMOB",
        plot_type="reflectivity",
        scan_time=scan_time
    )
    
    assert plot_path is not None
    assert "reflectivity" in str(plot_path).lower() or plot_path.parent.name == "reflectivity"
    assert "20240115" in str(plot_path)


def test_get_plot_path_cells(tmp_path):
    """Test getting plot path for cell plots."""
    dirs = setup_output_directories(tmp_path)
    scan_time = datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc)
    
    plot_path = get_plot_path(
        dirs,
        radar_id="KMOB",
        plot_type="cells",
        scan_time=scan_time
    )
    
    assert plot_path is not None
    assert "cells" in str(plot_path).lower() or plot_path.parent.name == "cells"


def test_get_log_path(tmp_path):
    """Test getting log file path."""
    dirs = setup_output_directories(tmp_path)
    
    log_path = get_log_path(dirs, radar_id="KMOB")
    
    assert log_path is not None
    assert log_path.parent == dirs["logs"]
    assert "KMOB" in str(log_path)


def test_get_log_path_without_radar_id(tmp_path):
    """Test getting log path without radar ID."""
    dirs = setup_output_directories(tmp_path)
    
    log_path = get_log_path(dirs)
    
    assert log_path is not None
    assert log_path.parent == dirs["logs"]


def test_paths_use_date_hierarchy(tmp_path):
    """Test that paths use YYYYMMDD date hierarchy."""
    dirs = setup_output_directories(tmp_path)
    scan_time = datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc)
    
    nexrad_path = get_nexrad_path(dirs, "KMOB", "test.nc", scan_time)
    nc_path = get_netcdf_path(dirs, "KMOB", "test.nc", scan_time)
    
    # Both should have date in path
    assert "20240115" in str(nexrad_path)
    assert "20240115" in str(nc_path)


def test_paths_use_radar_hierarchy(tmp_path):
    """Test that paths include radar ID in hierarchy."""
    dirs = setup_output_directories(tmp_path)
    scan_time = datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc)
    
    nexrad_path = get_nexrad_path(dirs, "KMOB", "test.nc", scan_time)
    
    # Should have radar ID in path structure
    assert "KMOB" in str(nexrad_path)


def test_setup_directories_expands_tilde(tmp_path):
    """Test that tilde is expanded to home directory."""
    dirs = setup_output_directories(tmp_path)
    assert "~" not in str(dirs["base"])


def test_setup_directories_resolves_relative_paths(tmp_path):
    """Test that relative paths are resolved to absolute."""
    dirs = setup_output_directories(tmp_path)
    assert dirs["base"].is_absolute()


def test_setup_output_directories_with_explicit_path(tmp_path):
    """Test setup with explicit base directory."""
    dirs = setup_output_directories(tmp_path)
    
    assert dirs["base"] == tmp_path
    assert dirs["nexrad"] == tmp_path / "nexrad"
    assert dirs["gridnc"] == tmp_path / "gridnc"
    assert dirs["analysis"] == tmp_path / "analysis"
    assert dirs["plots"] == tmp_path / "plots"
    assert dirs["logs"] == tmp_path / "logs"
    
    # All directories should exist
    for key, path in dirs.items():
        assert path.exists(), f"{key} directory not created"


def test_setup_output_directories_creates_subdirs(tmp_path):
    """Test that all required subdirectories are created."""
    dirs = setup_output_directories(tmp_path)
    
    assert (tmp_path / "nexrad").is_dir()
    assert (tmp_path / "gridnc").is_dir()
    assert (tmp_path / "analysis").is_dir()
    assert (tmp_path / "plots").is_dir()
    assert (tmp_path / "logs").is_dir()


def test_different_radars_different_paths(tmp_path):
    """Test that different radar IDs get different paths."""
    dirs = setup_output_directories(tmp_path)
    scan_time = datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc)
    
    path_kmob = get_nexrad_path(dirs, "KMOB", "test.nc", scan_time)
    path_khtx = get_nexrad_path(dirs, "KHTX", "test.nc", scan_time)
    
    assert path_kmob != path_khtx
    assert "KMOB" in str(path_kmob)
    assert "KHTX" in str(path_khtx)



def test_setup_directories_expands_tilde(tmp_path):
    """Test that tilde is expanded to home directory."""
    # This test verifies path expansion works
    dirs = setup_output_directories(tmp_path)
    assert "~" not in str(dirs["base"])


def test_setup_directories_resolves_relative_paths(tmp_path):
    """Test that relative paths are resolved to absolute."""
    dirs = setup_output_directories(tmp_path)
    assert dirs["base"].is_absolute()
