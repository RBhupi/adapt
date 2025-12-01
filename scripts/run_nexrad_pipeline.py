#!/usr/bin/env python3
"""ADAPT NEXRAD Radar Processing Pipeline Runner.

Usage:
    python scripts/run_nexrad_pipeline.py scripts/user_config.py
    python scripts/run_nexrad_pipeline.py scripts/user_config.py --radar KHTX
    python scripts/run_nexrad_pipeline.py scripts/user_config.py --mode historical

Note: User config in scripts/user_config.py, expert config in src/expert_config.py

Author: Bhupendra Raut
"""

import sys
import argparse
import importlib.util
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from adapt.setup_directories import setup_output_directories
from adapt.pipeline.orchestrator import PipelineOrchestrator
from expert_config import PIPELINE_CONFIG


def load_config(config_path: str) -> dict:
    """Load user config from Python file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    
    spec = importlib.util.spec_from_file_location("config_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find CONFIG dict
    for name in dir(module):
        if name.startswith('CONFIG'):
            obj = getattr(module, name)
            if isinstance(obj, dict):
                return obj
    
    raise ValueError(f"No CONFIG dict found in {path}")


def user_to_internal_config(user: dict) -> dict:
    """Convert user-facing config to internal format.
    
    Merges user settings with expert_config defaults.
    """
    # Start with expert config as base
    config = PIPELINE_CONFIG.copy()
    
    # Override with user settings
    config.update({
        "mode": user.get("MODE", "realtime"),
        "base_dir": user.get("BASE_DIR", "/tmp/adapt_output"),
        "reader": {
            "file_format": "nexrad_archive",
        },
        "downloader": {
            "radar_id": user.get("RADAR_ID", "KDIX"),
            "output_dir": None,  # Set later
            "latest_n": user.get("LATEST_FILES", 5),
            "minutes": user.get("LATEST_MINUTES", 60),
            "sleep_interval": user.get("POLL_INTERVAL_SEC", 300),
            "start_time": user.get("START_TIME"),
            "end_time": user.get("END_TIME"),
        },
        "regridder": {
            "grid_shape": user.get("GRID_SHAPE", (41, 301, 301)),
            "grid_limits": user.get("GRID_LIMITS", ((0, 20000), (-150000, 150000), (-150000, 150000))),
            "roi_func": "dist_beam",
            "min_radius": 1750.0,
            "weighting_function": "cressman",
            "save_netcdf": True,
        },
        "segmenter": {
            "method": user.get("SEGMENTATION_METHOD", "threshold"),
            "threshold": user.get("THRESHOLD_DBZ", 30),
            "min_cellsize_gridpoint": user.get("MIN_CELL_SIZE", 5),
            "max_cellsize_gridpoint": user.get("MAX_CELL_SIZE"),
            "closing_kernel": (2, 2),
            "filter_by_size": True,
        },
        "global": {
            "z_level": user.get("Z_LEVEL", 2000),
            "var_names": {
                "reflectivity": user.get("REFLECTIVITY_VAR", "reflectivity"),
                "cell_labels": "cell_labels",
            },
            "coord_names": {"time": "time", "z": "z", "y": "y", "x": "x"},
        },
        "projector": {
            **config.get("projector", {}),
            "method": user.get("PROJECTION_METHOD", "adapt_default"),
            "max_projection_steps": user.get("PROJECTION_STEPS", 5),
        },
    })
    
    # Note: visualization, analyzer, output, logging settings come from expert_config
    # unless overridden above
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Run ADAPT radar processing pipeline")
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("--radar", help="Override radar ID")
    parser.add_argument("--mode", choices=["realtime", "historical"], help="Override mode")
    parser.add_argument("--start", help="Start time (ISO format)")
    parser.add_argument("--end", help="End time (ISO format)")
    parser.add_argument("--max-runtime", type=int, help="Max runtime in minutes (realtime)")
    parser.add_argument("--rerun", action="store_true", help="Delete output directories before running")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()
    
    # Load and convert config
    user_config = load_config(args.config)
    config = user_to_internal_config(user_config)
    
    # Apply command-line overrides
    if args.radar:
        config["downloader"]["radar_id"] = args.radar
    if args.mode:
        config["mode"] = args.mode
    if args.start:
        config["downloader"]["start_time"] = args.start
    if args.end:
        config["downloader"]["end_time"] = args.end
    if args.verbose:
        config["logging"]["level"] = "DEBUG"
    
    # Clean output directories if --rerun specified
    if args.rerun:
        import shutil
        base_dir = Path(config["base_dir"])
        if base_dir.exists():
            print(f"🗑️  Cleaning output directory: {base_dir}")
            shutil.rmtree(base_dir)
            print("✓ Output directory cleaned")
    
    # Setup output directories
    output_dirs = setup_output_directories(config["base_dir"])
    config["output_dirs"] = output_dirs
    config["downloader"]["output_dir"] = str(output_dirs["nexrad"])
    
    # Print summary
    print(f"\n{'='*60}")
    print("ADAPT RADAR PROCESSING PIPELINE")
    print('='*60)
    print(f"Config: {args.config}")
    print(f"Radar:  {config['downloader']['radar_id']}")
    print(f"Mode:   {config['mode']}")
    print(f"Output: {config['base_dir']}")
    print('='*60)
    
    # Run pipeline
    orchestrator = PipelineOrchestrator(config)
    orchestrator.start(max_runtime=args.max_runtime)


if __name__ == "__main__":
    main()
