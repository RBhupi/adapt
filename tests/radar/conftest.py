# tests/conftest.py
from datetime import datetime, timezone
import pytest
import numpy as np
import xarray as xr

# ---- AwsNexradDownloader fixtures ----
class FakeScan:
    def __init__(self, key, scan_time=None):
        self.key = key
        self.scan_time = scan_time or datetime.now(timezone.utc)


class FakeAwsConn:
    def __init__(self, scans):
        self.scans = scans

    def get_avail_scans_in_range(self, start, end, radar_id):
        return self.scans

    def download(self, scans, target_dir, keep_aws_folders=False):
        class Result:
            def __init__(self, path):
                self.filepath = path

        results = []
        for scan in scans:
            path = target_dir / scan.key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * 2048)
            results.append(Result(path))

        class DownloadResults:
            def iter_success(self):
                return results

        return DownloadResults()


@pytest.fixture
def fake_scan():
    return FakeScan


@pytest.fixture
def fake_aws_conn():
    return FakeAwsConn


# ---- RadarCellSegmenter fixtures ----
# these are for non-closing tests, so default kernel size of (1,1) is  used.
@pytest.fixture
def simple_2d_ds():
    """
    2D reflectivity field with one clear cell.
    """
    data = np.array([
        [10, 10, 10, 10],
        [10, 40, 40, 10],
        [10, 40, 40, 10],
        [10, 10, 10, 10],
    ], dtype=np.float32)

    ds = xr.Dataset(
        {
            "reflectivity": (("y", "x"), data)
        },
        coords={
            "y": np.arange(data.shape[0]),
            "x": np.arange(data.shape[1]),
        },
        attrs={"z_level_m": 2000},
    )
    return ds


@pytest.fixture
def empty_2d_ds():
    """
    All values below threshold, so no cells.
    """
    data = np.zeros((4, 4), dtype=np.float32)

    return xr.Dataset(
        {"reflectivity": (("y", "x"), data)},
        coords={"y": range(4), "x": range(4)},
        attrs={"z_level_m": 1000},
    )


@pytest.fixture
def two_cell_ds():
    """
    Two separate cells of different sizes.
    """
    data = np.array([
        [50, 50,  0,  0,  0],
        [50, 50,  0, 30, 30],
        [ 0,  0,  0, 30, 30],
        [ 0,  0,  0,  0,  0],
    ], dtype=np.float32)

    return xr.Dataset(
        {"reflectivity": (("y", "x"), data)},
        coords={"y": range(4), "x": range(5)},
    )


# This is fo testing segmentation with multiple cells and closing operations
@pytest.fixture
def large_multi_cell_ds():
    """
    Larger domain with multiple well-separated cells.
    No closing should keep all separate.
    """
    data = np.zeros((10, 10), dtype=np.float32)

    # Cell 1 (top-left)
    data[1:3, 1:3] = 45

    # Cell 2 (top-right)
    data[1:3, 7:9] = 50

    # Cell 3 (bottom-left)
    data[7:9, 1:3] = 55

    # Cell 4 (bottom-right)
    data[7:9, 7:9] = 60

    return xr.Dataset(
        {"reflectivity": (("y", "x"), data)},
        coords={"y": range(10), "x": range(10)},
        attrs={"z_level_m": 2000},
    )


@pytest.fixture
def close_cells_ds():
    """
    Two nearby cells separated by a 1-pixel gap.
    Closing (2,2) should merge them.
    """
    data = np.zeros((6, 6), dtype=np.float32)

    # Cell A
    data[2:4, 1:3] = 40

    # 1-pixel gap

    # Cell B
    data[2:4, 4:6] = 40

    return xr.Dataset(
        {"reflectivity": (("y", "x"), data)},
        coords={"y": range(6), "x": range(6)},
    )


# For testing motion projection

@pytest.fixture
def simple_labeled_ds_pair():
    """
    Two small 2D datasets with:
    - reflectivity
    - cell_labels
    - valid time coordinate
    Zero motion between frames.
    """
    data = np.array([
        [0, 40, 40, 0],
        [0, 40, 40, 0],
        [0,  0,  0, 0],
        [0,  0,  0, 0],
    ], dtype=np.float32)

    labels = np.array([
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)

    t0 = np.datetime64("2024-01-01T00:00")
    t1 = np.datetime64("2024-01-01T00:05")

    ds1 = xr.Dataset(
        {
            "reflectivity": (("y", "x"), data),
            "cell_labels": (("y", "x"), labels),
        },
        coords={"y": range(4), "x": range(4)},
    )
    ds1 = ds1.assign_coords(time=t0)

    ds2 = xr.Dataset(
        {
            "reflectivity": (("y", "x"), data),
            "cell_labels": (("y", "x"), labels),
        },
        coords={"y": range(4), "x": range(4)},
    )
    ds2 = ds2.assign_coords(time=t1)

    return [ds1, ds2]


# Analyzer fixtures
@pytest.fixture
def labeled_ds_with_extras(simple_2d_ds):
    """
    2D dataset with:
    - cell_labels
    - reflectivity
    - heading vectors
    - projections
    """
    ds = simple_2d_ds.copy()

    labels = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)

    ds["cell_labels"] = (("y", "x"), labels)

    ds["heading_x"] = (("y", "x"), np.ones_like(labels, dtype=np.float32))
    ds["heading_y"] = (("y", "x"), np.zeros_like(labels, dtype=np.float32))

    projections = np.stack([labels, labels], axis=0)
    ds["cell_projections"] = (
        ("frame_offset", "y", "x"),
        projections
    )

    ds = ds.assign_coords(frame_offset=[0, 1])
    ds = ds.assign_coords(time=np.datetime64("2024-01-01T00:00"))

    return ds


@pytest.fixture
def basic_config(temp_dir):
    config = {
    "mode": "realtime",
    
    "reader": {
        "file_format": "nexrad_archive",
    },
    
    "downloader": {
        "radar_id": None,
        "output_dir": None,
        "latest_n": 5,
        "minutes": 60,
        "sleep_interval": 300,
        "start_time": None,
        "end_time": None,
    },
    
    "regridder": {
        "grid_shape": (41, 301, 301),
        "grid_limits": ((0, 20000), (-150000, 150000), (-150000, 150000)),
        "roi_func": "dist_beam",
        "min_radius": 1750.0,
        "weighting_function": "cressman",
        "save_netcdf": True,
    },
    
    "segmenter": {
        "method": "threshold",
        "threshold": 30,
        "min_cellsize_gridpoint": 5,
        "max_cellsize_gridpoint": None,
        "closing_kernel": (2, 2),
        "filter_by_size": True,
    },
    
    "global": {
        "z_level": 2000,
        "var_names": {
            "reflectivity": "reflectivity",
            "cell_labels": "cell_labels",
        },
        "coord_names": {
            "time": "time",
            "z": "z",
            "y": "y",
            "x": "x",
        },
    },
    
    "projector": {
        "method": "adapt_default",
        "max_time_interval_minutes": 30,
        "max_projection_steps": 5,
        "nan_fill_value": 0,
        "flow_params": {
            "pyr_scale": 0.5,
            "levels": 3,  # Pyramid levels
            "winsize": 21,  # Window size
            "iterations": 10,  # Number of iterations
            "poly_n": 7,  # Polynomial expansion degree
            "poly_sigma": 1.5,  # Gaussian sigma
            "flags": 0,
        },
        "min_motion_threshold": 0.5,  # Minimum motion threshold
    },
    
    "analyzer": {
        # Whitelist of radar measurement variables to analyze
        # Only these variables will have statistics extracted and saved to database
        "radar_variables": [
            "reflectivity",
            "velocity",
            "differential_phase",
            "differential_reflectivity",
            "spectrum_width",
            "cross_correlation_ratio",
        ],
        # Fields to always exclude (metadata, segmentation, flow vectors)
        "exclude_fields": [
            "ROI", "labels", "cell_labels",  # Segmentation metadata
            "cell_projections",  # Projection metadata
            "clutter_filter_power_removed"  # Clutter filter
        ],
    },
    
    "visualization": {
        "enabled": True,
        "dpi": 200,
        "figsize": (18, 8),  # Width x Height in inches
        "output_format": "png",  # png, pdf, or jpeg
        "use_basemap": True,
        "basemap_alpha": 0.6,
        
        # Contour styling
        "seg_linewidth": 0.8,  # Segmentation contour width
        "proj_linewidth": 1.0,  # Projection contour width
        "proj_alpha": 0.8,  # Projection contour transparency
        
        # Flow vector styling
        "flow_scale": 0.5,  # Lower = larger arrows (15-40 for long arrows)
        "flow_subsample": 10,  # Sample every Nth pixel
        
        # Reflectivity thresholds
        "min_reflectivity": 10,
        "refl_vmin": 10,
        "refl_vmax": 50,
    },
    
    "output": {
        "compression": "snappy",
    },
    
    "logging": {
        "level": "INFO",
    },
}

    return config


import tempfile
import shutil
from pathlib import Path
@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)



