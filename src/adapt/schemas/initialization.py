"""Complete runtime initialization for ADAPT pipeline.

This module handles ALL initialization responsibilities:
- Configuration resolution (CLI > User > Param)  
- Output directory setup
- Cleanup handling (--rerun)
- Configuration persistence with run ID
- Returns fully ready InternalConfig for orchestrator

Author: Bhupendra Raut
"""

import importlib.util
import shutil
import json
from pathlib import Path
from typing import Dict
from datetime import datetime, timezone

from adapt.schemas.resolve import resolve_config
from adapt.schemas.param import ParamConfig
from adapt.schemas.user import UserConfig 
from adapt.schemas.cli import CLIConfig
from adapt.schemas.internal import InternalConfig
from adapt.core import DataRepository


def _load_user_config_dict(config_path: str) -> dict:
    """Load user config dict from Python file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    
    spec = importlib.util.spec_from_file_location("config_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config module from {path}")
        
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find CONFIG dict
    for name in dir(module):
        if name.startswith('CONFIG'):
            obj = getattr(module, name)
            if isinstance(obj, dict):
                return obj
    
    raise ValueError(f"No CONFIG dict found in {path}")


def _setup_output_directories(base_dir: str) -> Dict[str, Path]:
    """Setup output directory structure.
    
    Creates the standard ADAPT directory layout under base_dir.
    This was previously in setup_directories module but belongs here.
    """
    from adapt.setup_directories import setup_output_directories
    return setup_output_directories(base_dir)


def _handle_rerun_cleanup(base_dir: str, rerun: bool) -> None:
    """Handle --rerun directory cleanup if requested."""
    if not rerun:
        return
        
    base_dir_path = Path(base_dir)
    if base_dir_path.exists():
        print(f"Cleaning output directory: {base_dir_path}")
        shutil.rmtree(base_dir_path)
        print("Output directory cleaned")


def _persist_runtime_config(config: InternalConfig, run_id: str, output_dirs: Dict[str, Path]) -> None:
    """Persist final runtime configuration to output directory with run ID.
    
    Saves the complete resolved configuration for reproducibility and debugging.
    """
    config_output_dir = Path(output_dirs["base"])
    config_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config with run ID in filename  
    config_file = config_output_dir / f"runtime_config_{run_id}.json"
    
    # Add run_id to config dict for persistence
    config_dict = config.model_dump()
    config_dict["run_id"] = run_id
    config_dict["created_at"] = datetime.now(timezone.utc).isoformat()
    
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    print(f"Runtime config saved: {config_file}")


def init_runtime_config(args) -> InternalConfig:
    """Complete runtime initialization - single entry point for ADAPT.
    
    Handles ALL initialization responsibilities:
    1. Configuration resolution (CLI > User > Param)
    2. Cleanup handling (--rerun)  
    3. Output directory setup
    4. Configuration persistence with run ID
    5. Returns fully ready InternalConfig for orchestrator
    
    This is the ONLY function user scripts should call from schemas.
    Everything else is internal implementation.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments with config path and all overrides
        
    Returns
    -------
    InternalConfig
        Fully validated, ready-to-use runtime configuration with:
        - All directories created and paths set
        - All CLI overrides applied
        - Run ID generated and included
        - Configuration persisted to output directory
        
    Examples
    --------
    >>> args = parser.parse_args()
    >>> config = init_runtime_config(args) 
    >>> orchestrator = PipelineOrchestrator(config)
    """
    # 1. Load and resolve configuration from all sources
    config_path = getattr(args, 'config', None)
    if not config_path:
        raise ValueError("Config path required in args.config")
    
    # Load components
    param_cfg = ParamConfig()
    user_cfg_dict = _load_user_config_dict(config_path)
    user_cfg = UserConfig.model_validate(user_cfg_dict)
    
    # Create CLI config from args  
    cli_args = {
        k: v
        for k, v in {
            "radar_id": getattr(args, 'radar_id', None),
            "mode": getattr(args, 'mode', None), 
            "start_time": getattr(args, 'start_time', None),
            "end_time": getattr(args, 'end_time', None),
            "base_dir": getattr(args, 'base_dir', None),
            "log_level": "DEBUG" if getattr(args, 'verbose', False) else None,
        }.items()
        if v is not None
    }
    cli_cfg = CLIConfig.model_validate(cli_args)
    
    # Resolve to final internal config
    internal_config_dict = resolve_config(param_cfg, user_cfg, cli_cfg).model_dump()
    
    # 2. Handle --rerun cleanup BEFORE directory setup
    rerun = getattr(args, 'rerun', False)
    _handle_rerun_cleanup(internal_config_dict["base_dir"], rerun)
    
    # 3. Setup output directories  
    output_dirs = _setup_output_directories(internal_config_dict["base_dir"])
    
    # Add output_dirs to config for orchestrator use
    internal_config_dict["output_dirs"] = {k: str(v) for k, v in output_dirs.items()}
    
    # 4. Generate run ID and create final config
    run_id = DataRepository.generate_run_id()
    internal_config_dict["run_id"] = run_id
    
    config = InternalConfig.model_validate(internal_config_dict)
    
    # 5. Persist configuration for reproducibility
    _persist_runtime_config(config, run_id, output_dirs)
    
    print(f"Runtime initialization complete. Run ID: {run_id}")
    
    return config


# Only this function is exposed - everything else is internal
__all__ = ['init_runtime_config']