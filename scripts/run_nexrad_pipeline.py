#!/usr/bin/env python3
"""``Adapt`` NEXRAD Radar Processing Pipeline Runner.

THIN WRAPPER around adapt.cli.run_nexrad.run_nexrad_pipeline().
All business logic lives in src/adapt/cli/ - this script only parses arguments.

Usage:
    python scripts/run_nexrad_pipeline.py scripts/user_config.py
    python scripts/run_nexrad_pipeline.py scripts/user_config.py --radar-id KHTX
    python scripts/run_nexrad_pipeline.py scripts/user_config.py --mode historical

Note: User config in scripts/user_config.py, expert config in src/param_config.py

Author: Bhupendra Raut
"""

import sys
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from adapt.cli import run_nexrad_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run the ADAPT radar processing pipeline")
    parser.add_argument("config", help="Path to user config file")
    parser.add_argument("--radar-id", help="Override radar ID")
    parser.add_argument("--mode", choices=["realtime", "historical"], help="Override mode")
    parser.add_argument("--start-time", help="Start time (ISO format)")
    parser.add_argument("--end-time", help="End time (ISO format)")
    parser.add_argument("--base-dir", help="Output directory")
    parser.add_argument("--max-runtime", type=int, help="Max runtime in minutes (realtime)")
    parser.add_argument("--rerun", action="store_true", help="Delete output directories before running")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()
    
    # Build CLI args dict
    cli_args = {
        "radar_id": args.radar_id,
        "mode": args.mode,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "base_dir": args.base_dir,
        "log_level": "DEBUG" if args.verbose else None,
    }
    
    # Call core pipeline runner (all logic is there)
    run_nexrad_pipeline(
        user_config_path=args.config,
        cli_args=cli_args,
        max_runtime=args.max_runtime,
        rerun=args.rerun,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
