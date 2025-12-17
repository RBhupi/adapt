"""Pydantic configuration schemas for ADAPT pipeline.

This module provides strictly typed configuration models for the ADAPT
radar processing pipeline. All configuration validation, coercion, and
normalization happens at schema validation time via Pydantic.

Exports
-------
resolve_config : function
    Single entrypoint for configuration resolution
InternalConfig : class
    Fully validated, authoritative runtime configuration
ParamConfig : class
    Expert defaults (complete)
UserConfig : class
    User-facing configuration (forgiving, minimal)
CLIConfig : class
    Command-line operational overrides
"""

from adapt.schemas.resolve import resolve_config
from adapt.schemas.internal import InternalConfig
from adapt.schemas.param import ParamConfig
from adapt.schemas.user import UserConfig
from adapt.schemas.cli import CLIConfig

__all__ = [
    'resolve_config',
    'InternalConfig',
    'ParamConfig',
    'UserConfig',
    'CLIConfig',
]
