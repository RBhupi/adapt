#!/usr/bin/env python3
"""``Adapt`` NEXRAD Radar Processing Pipeline Runner.

Usage:
    python scripts/run_nexrad_pipeline.py scripts/user_config.py
    python scripts/run_nexrad_pipeline.py scripts/user_config.py --radar-id KHTX
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
    parser.add_argument("--radar-id", help="Override radar ID")
    parser.add_argument("--mode", choices=["realtime", "historical"], help="Override mode")
    parser.add_argument("--start-time", help="Start time (ISO format)")
    parser.add_argument("--end-time", help="End time (ISO format)")
    parser.add_argument("--base-dir", help="Output directory")
    parser.add_argument("--max-runtime", type=int, help="Max runtime in minutes (realtime)")
    parser.add_argument("--rerun", action="store_true", help="Delete output directories before running")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()
    
    # Load configurations
    param_cfg = ParamConfig()  # Expert defaults
    
    # Load user config from file (pass raw dict directly to UserConfig)
    user_cfg_dict = load_user_config_dict(args.config)
    # UserConfig performs normalization/validation internally (including
    # legacy uppercase keys). Do NOT mutate the resulting object.
    user_cfg = UserConfig.model_validate(user_cfg_dict)
    
    # Create CLI config from command-line args (CLIConfig owns overrides)
    cli_cfg = CLIConfig.model_validate({
        k: v
        for k, v in {
            "radar_id": args.radar_id,
            "mode": args.mode,
            "start_time": args.start_time,
            "end_time": args.end_time,
            "base_dir": args.base_dir,
            "log_level": "DEBUG" if args.verbose else None,
        }.items()
        if v is not None
    })
    
    # Resolve to internal config (Param < User < CLI)
    config = resolve_config(param_cfg, user_cfg, cli_cfg)
    
    # Clean output directories if --rerun specified
    if args.rerun:
        import shutil
        base_dir_path = Path(config.base_dir)
        if base_dir_path.exists():
            print(f"Cleaning output directory: {base_dir_path}")
            shutil.rmtree(base_dir_path)
            print("Output directory cleaned")
    
    # Setup output directories
    output_dirs = setup_output_directories(config.base_dir)
    
    # Print summary
    print(f"\n{'='*60}")
    print("ADAPT Radar Processing Pipeline")
    print('='*60)
    print(f"Config: {args.config}")
    print(f"Radar:  {config.downloader.radar_id}")
    print(f"Mode:   {config.mode}")
    print(f"Output: {config.base_dir}")
    print('='*60)

    if args.verbose:
        import json
        print("\nFull Internal Configuration:")
        # Use model_dump_json for pretty printing, or just model_dump
        print(json.dumps(config.model_dump(), indent=2))
        print('='*60)
    
    # Run pipeline
    orchestrator = PipelineOrchestrator(config, output_dirs)
    orchestrator.start(max_runtime=args.max_runtime)


if __name__ == "__main__":
    main()
