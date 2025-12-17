"""CLIConfig: Command-line operational overrides.

Minimal configuration for operational parameters that commonly change
between runs: mode, radar ID, output paths, verbosity.

This schema handles command-line arguments parsed by argparse.
"""

from typing import Literal, Optional
from pydantic import Field
from adapt.schemas.base import AdaptBaseModel


class CLIConfig(AdaptBaseModel):
    """Command-line configuration overrides.
    
    Operational-only settings that override user and param configs.
    Highest priority in config resolution.
    
    Usage
    -----
        cli_cfg = CLIConfig(
            mode="historical",
            radar_id="KHTX",
            base_dir="/scratch/adapt_output",
        )
        
        internal = resolve_config(param_cfg, user_cfg, cli_cfg)
    """
    
    mode: Optional[Literal["realtime", "historical"]] = None
    radar_id: Optional[str] = None
    base_dir: Optional[str] = None
    log_level: Optional[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]] = None
    
    def to_internal_overrides(self) -> dict:
        """Convert CLI config to internal config structure.
        
        Returns
        -------
        dict
            Nested dictionary matching InternalConfig structure
        """
        overrides = {}
        
        if self.mode is not None:
            overrides["mode"] = self.mode
        
        if self.radar_id is not None:
            overrides["downloader"] = {"radar_id": self.radar_id}
        
        if self.log_level is not None:
            overrides["logging"] = {"level": self.log_level}
        
        # base_dir handled separately by setup_output_directories
        
        return overrides
