""" ``Adapt`` User Configuration.

This is the user-facing configuration file. Modify settings here to customize
the pipeline behavior. this overrides internal defaults defined in src/param_config.py
"""

CONFIG = {
    # ========================================================================
    # PIPELINE MODE & TARGET
    # ========================================================================
    "mode": "realtime",       # "realtime" or "historical"
    "radar_id": "KJGX",       # NWS NEXRAD radar ID (e.g., KDIX, KHTX, KLOT)
    "base_dir": "/Users/bhupendra/projects/arm_radar_scaning/tmp",  # All outputs go here

    # ========================================================================
    # REALTIME MODE SETTINGS
    # ========================================================================
    "latest_files": 5,        # Number of latest files to keep
    "latest_minutes": 60,     # Time window in minutes
    "poll_interval_sec": 30,  # Seconds between AWS polls

    # ========================================================================
    # HISTORICAL MODE SETTINGS
    # ========================================================================
    "start_time": None,       # ISO format: "2025-03-05T15:00:00Z"
    "end_time": None,         # ISO format: "2025-03-05T18:00:00Z"

    # ========================================================================
    # GRID SETTINGS
    # ========================================================================
    "grid_shape": (41, 301, 301),  # (z, y, x) grid points
    "grid_limits": (
        (0, 20000),           # z: 0-20km altitude
        (-150000, 150000),    # y: ±150km
        (-150000, 150000),    # x: ±150km
    ),

    # ========================================================================
    # SEGMENTATION SETTINGS
    # ========================================================================
    "z_level": 2000,          # Analysis altitude in meters
    "reflectivity_var": "reflectivity",
    "segmentation_method": "threshold",
    "threshold": 30,          # Cell detection threshold
    "min_cellsize_gridpoint": 5,       # Minimum cell size (grid points)
    "max_cellsize_gridpoint": None,    # Maximum cell size (None = no limit)    
    # ========================================================================
    # PROJECTION SETTINGS
    # ========================================================================
    "projection_method": "adapt_default",
    "max_projection_steps": 5,
    # Note: Advanced optical flow parameters (pyramid levels, window size, etc.)
    # are configured in src/expert_config.py
}
