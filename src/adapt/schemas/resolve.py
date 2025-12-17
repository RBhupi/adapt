"""Configuration resolution and merging logic.

This module provides the single entrypoint for configuration resolution:
resolve_config(). It merges ParamConfig, UserConfig, and CLIConfig in
the correct precedence order and returns a validated InternalConfig.

Precedence (highest to lowest):
1. CLIConfig (command-line overrides)
2. UserConfig (user file)
3. ParamConfig (expert defaults)
"""

from typing import Union, Optional, Any
from adapt.schemas.param import ParamConfig
from adapt.schemas.user import UserConfig
from adapt.schemas.cli import CLIConfig
from adapt.schemas.internal import InternalConfig


def deep_merge(base: dict, *overrides: dict) -> dict:
    """Deep merge multiple dictionaries.
    
    Later dictionaries override earlier ones. Nested dictionaries are
    merged recursively; other values are replaced.
    
    Parameters
    ----------
    base : dict
        Base dictionary (lowest priority)
    *overrides : dict
        Override dictionaries (higher priority, left to right)
    
    Returns
    -------
    dict
        Merged dictionary
    
    Examples
    --------
    >>> base = {"a": 1, "b": {"c": 2, "d": 3}}
    >>> override = {"b": {"d": 4, "e": 5}, "f": 6}
    >>> deep_merge(base, override)
    {'a': 1, 'b': {'c': 2, 'd': 4, 'e': 5}, 'f': 6}
    """
    result = base.copy()
    
    for override in overrides:
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursive merge for nested dicts
                result[key] = deep_merge(result[key], value)
            else:
                # Replace value
                result[key] = value
    
    return result


def resolve_config(
    param_cfg: Union[dict, ParamConfig],
    user_cfg: Optional[Union[dict, UserConfig]] = None,
    cli_cfg: Optional[Union[dict, CLIConfig]] = None,
) -> InternalConfig:
    """Resolve final runtime configuration from param, user, and CLI configs.
    
    This is the SINGLE ENTRYPOINT for configuration resolution. It validates
    and merges configs in the correct precedence order, then returns an
    immutable InternalConfig for runtime use.
    
    Precedence (highest to lowest):
    1. CLIConfig (command-line overrides)
    2. UserConfig (user file overrides)
    3. ParamConfig (expert defaults)
    
    Parameters
    ----------
    param_cfg : dict or ParamConfig
        Expert configuration with complete defaults. Required.
    user_cfg : dict or UserConfig, optional
        User configuration with overrides. If None or empty, uses only param defaults.
    cli_cfg : dict or CLIConfig, optional
        Command-line overrides. If None or empty, no CLI overrides applied.
    
    Returns
    -------
    InternalConfig
        Fully validated, immutable runtime configuration
    
    Raises
    ------
    ValidationError
        If any config fails Pydantic validation
    
    Examples
    --------
    >>> from adapt.schemas import resolve_config, ParamConfig, UserConfig
    >>> 
    >>> # Load expert defaults
    >>> param = ParamConfig()
    >>> 
    >>> # User wants higher threshold
    >>> user = UserConfig(threshold_dbz=35, radar_id="KHTX")
    >>> 
    >>> # Resolve to internal config
    >>> config = resolve_config(param, user)
    >>> config.segmenter.threshold
    35.0
    >>> config.downloader.radar_id
    'KHTX'
    """
    # Validate/convert inputs to Pydantic models
    if not isinstance(param_cfg, ParamConfig):
        param = ParamConfig.model_validate(param_cfg)
    else:
        param = param_cfg
    
    if user_cfg is None or (isinstance(user_cfg, dict) and not user_cfg):
        user = UserConfig()
    elif not isinstance(user_cfg, UserConfig):
        user = UserConfig.model_validate(user_cfg)
    else:
        user = user_cfg
    
    if cli_cfg is None or (isinstance(cli_cfg, dict) and not cli_cfg):
        cli = CLIConfig()
    elif not isinstance(cli_cfg, CLIConfig):
        cli = CLIConfig.model_validate(cli_cfg)
    else:
        cli = cli_cfg
    
    # Convert to dicts for merging
    param_dict = param.model_dump(by_alias=True)  # Use 'global' not 'global_'
    user_overrides = user.to_internal_overrides()
    cli_overrides = cli.to_internal_overrides()
    
    # Deep merge: param < user < cli
    merged = deep_merge(param_dict, user_overrides, cli_overrides)
    
    # Cap max_projection_steps at 10
    if "projector" in merged and "max_projection_steps" in merged["projector"]:
        merged["projector"]["max_projection_steps"] = min(
            merged["projector"]["max_projection_steps"], 10
        )
    
    # Validate and freeze as InternalConfig
    internal = InternalConfig.model_validate(merged)
    
    return internal
