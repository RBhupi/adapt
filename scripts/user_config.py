"""ADAPT User Configuration.

This is the user-facing configuration file. Modify settings here to customize
the pipeline behavior. Advanced settings are in src/expert_config.py

Usage:
    python scripts/run_nexrad_pipeline.py scripts/user_config.py
    python scripts/run_nexrad_pipeline.py scripts/user_config.py --radar KHTX
    python scripts/run_nexrad_pipeline.py scripts/user_config.py --mode historical

Author: Bhupendra Raut
"""

CONFIG = {
    # ========================================================================
    # PIPELINE MODE & TARGET
    # ========================================================================
    "MODE": "realtime",       # "realtime" or "historical"
    "RADAR_ID": "KJGX",       # NWS NEXRAD radar ID (e.g., KDIX, KHTX, KLOT)
    "BASE_DIR": "/Users/bhupendra/projects/arm_radar_scaning/tmp",  # All outputs go here

    # ========================================================================
    # REALTIME MODE SETTINGS
    # ========================================================================
    "LATEST_FILES": 5,        # Number of latest files to keep
    "LATEST_MINUTES": 60,     # Time window in minutes
    "POLL_INTERVAL_SEC": 30,  # Seconds between AWS polls

    # ========================================================================
    # HISTORICAL MODE SETTINGS
    # ========================================================================
    "START_TIME": None,       # ISO format: "2025-03-05T15:00:00Z"
    "END_TIME": None,         # ISO format: "2025-03-05T18:00:00Z"

    # ========================================================================
    # GRID SETTINGS
    # ========================================================================
    "GRID_SHAPE": (41, 301, 301),  # (z, y, x) grid points
    "GRID_LIMITS": (
        (0, 20000),           # z: 0-20km altitude
        (-150000, 150000),    # y: ±150km
        (-150000, 150000),    # x: ±150km
    ),

    # ========================================================================
    # SEGMENTATION SETTINGS
    # ========================================================================
    "Z_LEVEL": 2000,          # Analysis altitude in meters
    "REFLECTIVITY_VAR": "reflectivity",
    "SEGMENTATION_METHOD": "threshold",
    "THRESHOLD_DBZ": 30,      # Cell detection threshold
    "MIN_CELL_SIZE": 5,       # Minimum cell size (grid points)
    "MAX_CELL_SIZE": None,    # Maximum cell size (None = no limit)    
    # ========================================================================
    # PROJECTION SETTINGS
    # ========================================================================
    "PROJECTION_METHOD": "adapt_default",
    "PROJECTION_STEPS": 5,
    # Note: Advanced optical flow parameters (pyramid levels, window size, etc.)
    # are configured in src/expert_config.py
}
