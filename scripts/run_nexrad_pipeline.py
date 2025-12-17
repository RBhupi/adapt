#!/usr/bin/env python3
"""``Adapt`` NEXRAD Radar Processing Pipeline Runner.

Usage:
    python scripts/run_nexrad_pipeline.py scripts/user_config.py
    python scripts/run_nexrad_pipeline.py scripts/user_config.py --radar KHTX
    python scripts/run_nexrad_pipeline.py scripts/user_config.py --mode historical

Note: User config in scripts/user_config.py, expert config in src/param_config.py

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
from adapt.schemas import resolve_config, ParamConfig, UserConfig, CLIConfig


def load_user_config_dict(config_path: str) -> dict:
    """Load user config dict from Python file.
    
    Returns the raw dict before Pydantic validation.
    """
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


def main():
    parser = argparse.ArgumentParser(description="Run the ADAPT radar processing pipeline")
    parser.add_argument("config", help="Path to user config file")
    parser.add_argument("--radar", help="Override radar ID")
    parser.add_argument("--mode", choices=["realtime", "historical"], help="Override mode")
    parser.add_argument("--start", help="Start time (ISO format)")
    parser.add_argument("--end", help="End time (ISO format)")
    parser.add_argument("--outdir", help="Output directory")
    parser.add_argument("--max-runtime", type=int, help="Max runtime in minutes (realtime)")
    parser.add_argument("--rerun", action="store_true", help="Delete output directories before running")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()
    
    # Load configurations
    param_cfg = ParamConfig()  # Expert defaults
    
    # Load user config from file
    user_cfg_dict = load_user_config_dict(args.config)
    # Convert uppercase keys to UserConfig-compatible format
    user_cfg_dict_normalized = {
        "mode": user_cfg_dict.get("MODE"),
        "radar_id": user_cfg_dict.get("RADAR_ID"),
        "base_dir": user_cfg_dict.get("BASE_DIR"),
        "latest_files": user_cfg_dict.get("LATEST_FILES"),
        "latest_minutes": user_cfg_dict.get("LATEST_MINUTES"),
        "poll_interval_sec": user_cfg_dict.get("POLL_INTERVAL_SEC"),
        "start_time": user_cfg_dict.get("START_TIME"),
        "end_time": user_cfg_dict.get("END_TIME"),
        "grid_shape": user_cfg_dict.get("GRID_SHAPE"),
        "grid_limits": user_cfg_dict.get("GRID_LIMITS"),
        "z_level": user_cfg_dict.get("Z_LEVEL"),
        "reflectivity_var": user_cfg_dict.get("REFLECTIVITY_VAR"),
        "segmentation_method": user_cfg_dict.get("SEGMENTATION_METHOD"),
        "threshold_dbz": user_cfg_dict.get("THRESHOLD_DBZ"),
        "min_cell_size": user_cfg_dict.get("MIN_CELL_SIZE"),
        "max_cell_size": user_cfg_dict.get("MAX_CELL_SIZE"),
        "projection_method": user_cfg_dict.get("PROJECTION_METHOD"),
        "projection_steps": user_cfg_dict.get("PROJECTION_STEPS"),
    }
    # Remove None values
    user_cfg_dict_normalized = {k: v for k, v in user_cfg_dict_normalized.items() if v is not None}
    user_cfg = UserConfig(**user_cfg_dict_normalized)
    
    # Create CLI config from command-line args
    cli_cfg_dict = {}
    if args.radar:
        cli_cfg_dict["radar_id"] = args.radar
    if args.mode:
        cli_cfg_dict["mode"] = args.mode
    if args.verbose:
        cli_cfg_dict["log_level"] = "DEBUG"
    cli_cfg = CLIConfig(**cli_cfg_dict)
    
    # Store base_dir separately (for directory setup)
    base_dir = args.outdir or user_cfg.base_dir or "/tmp/adapt_output"
    
    # Apply start/end time overrides if provided
    if args.start or args.end:
        # These override user config before resolution
        if args.start:
            user_cfg_dict_normalized["start_time"] = args.start
        if args.end:
            user_cfg_dict_normalized["end_time"] = args.end
        user_cfg = UserConfig(**user_cfg_dict_normalized)
    
    # Resolve to internal config
    config = resolve_config(param_cfg, user_cfg, cli_cfg)
    
    # Clean output directories if --rerun specified
    if args.rerun:
        import shutil
        base_dir_path = Path(base_dir)
        if base_dir_path.exists():
            print(f"🗑️  Cleaning output directory: {base_dir_path}")
            shutil.rmtree(base_dir_path)
            print("✓ Output directory cleaned")
    
    # Setup output directories
    output_dirs = setup_output_directories(base_dir)
    
    # Print summary
    print(f"\n{'='*60}")
    print("ADAPT Radar Processing Pipeline")
    print('='*60)
    print(f"Config: {args.config}")
    print(f"Radar:  {config.downloader.radar_id}")
    print(f"Mode:   {config.mode}")
    print(f"Output: {base_dir}")
    print('='*60)
    
    # Run pipeline
    orchestrator = PipelineOrchestrator(config, output_dirs)
    orchestrator.start(max_runtime=args.max_runtime)


if __name__ == "__main__":
    main()
