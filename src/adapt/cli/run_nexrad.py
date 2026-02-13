"""Core NEXRAD pipeline execution logic.

This module contains the actual pipeline runner, separated from argument parsing.
Scripts are thin wrappers; this is the real implementation.
"""

import sys
import json
import logging
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Any

from adapt.setup_directories import setup_output_directories
from adapt.pipeline.orchestrator import PipelineOrchestrator
from adapt.schemas import resolve_config, ParamConfig, UserConfig, CLIConfig


logger = logging.getLogger(__name__)


def load_user_config_dict(config_path: str) -> dict:
    """Load user config dict from Python file.
    
    Returns the raw dict before Pydantic validation.
    
    Parameters
    ----------
    config_path : str
        Path to user config Python file containing CONFIG dict.
        
    Returns
    -------
    dict
        Raw user configuration dictionary.
        
    Raises
    ------
    FileNotFoundError
        If config file does not exist.
    ValueError
        If no CONFIG dict found in file.
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


def run_nexrad_pipeline(
    user_config_path: str,
    cli_args: Optional[Dict[str, Any]] = None,
    max_runtime: Optional[int] = None,
    rerun: bool = False,
    verbose: bool = False
) -> None:
    """Execute the NEXRAD radar processing pipeline.
    
    This is the core pipeline execution function. It:
    1. Loads and resolves configuration (Param < User < CLI)
    2. Sets up output directories
    3. Optionally cleans directories if rerun=True
    4. Instantiates and starts the pipeline orchestrator
    5. Blocks until completion or interruption
    
    **No plotting, no visualization, no side effects beyond pipeline execution.**
    
    Parameters
    ----------
    user_config_path : str
        Path to user config file (Python file with CONFIG dict).
        
    cli_args : dict, optional
        CLI argument overrides. Keys: radar_id, mode, start_time, end_time,
        base_dir, log_level. All optional.
        
    max_runtime : int, optional
        Maximum runtime in minutes (realtime mode only).
        If None, runs until KeyboardInterrupt.
        
    rerun : bool, optional
        If True, delete output directories before running.
        
    verbose : bool, optional
        If True, enable DEBUG logging and print full resolved config.
        
    Raises
    ------
    FileNotFoundError
        If user_config_path does not exist.
    ValueError
        If configuration validation fails.
    KeyboardInterrupt
        If user presses Ctrl+C (pipeline stops gracefully).
        
    Examples
    --------
    Run with user config only::
    
        run_nexrad_pipeline("config/my_config.py")
        
    Run with CLI overrides::
    
        run_nexrad_pipeline(
            "config/my_config.py",
            cli_args={"radar_id": "KHTX", "mode": "realtime"},
            max_runtime=60
        )
        
    Run with verbose output::
    
        run_nexrad_pipeline(
            "config/my_config.py",
            verbose=True
        )
    """
    # Load configurations
    param_cfg = ParamConfig()  # Expert defaults
    
    # Load user config from file
    user_cfg_dict = load_user_config_dict(user_config_path)
    user_cfg = UserConfig.model_validate(user_cfg_dict)
    
    # Create CLI config from arguments
    cli_args = cli_args or {}
    if verbose and "log_level" not in cli_args:
        cli_args["log_level"] = "DEBUG"
    
    # Filter None values
    cli_dict = {k: v for k, v in cli_args.items() if v is not None}
    cli_cfg = CLIConfig.model_validate(cli_dict) if cli_dict else CLIConfig()
    
    # Resolve to internal config (Param < User < CLI)
    config = resolve_config(param_cfg, user_cfg, cli_cfg)
    
    # Clean output directories if --rerun specified
    if rerun:
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
    print(f"Config: {user_config_path}")
    print(f"Radar:  {config.downloader.radar_id}")
    print(f"Mode:   {config.mode}")
    print(f"Output: {config.base_dir}")
    print('='*60)

    if verbose:
        print("\nFull Internal Configuration:")
        print(json.dumps(config.model_dump(), indent=2))
        print('='*60)
    
    # Run pipeline (NO PLOTTING - orchestrator handles data production only)
    orchestrator = PipelineOrchestrator(config, output_dirs)
    orchestrator.start(max_runtime=max_runtime)
