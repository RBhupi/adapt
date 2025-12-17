"""UserConfig: Forgiving, minimal user-facing configuration.

This schema accepts user inputs in a variety of formats, with aliases
for common naming patterns (e.g., RADAR_ID → radar_id, MODE → mode).

UserConfig is intentionally minimal - users only specify what they want
to override from the expert defaults. Validation is lenient to accept
both uppercase and lowercase keys, integers where floats are expected, etc.
"""

from typing import Literal, Optional, Any
from pydantic import Field, field_validator, model_validator
from adapt.schemas.base import AdaptBaseModel


class UserSegmenterConfig(AdaptBaseModel):
    """User-facing segmentation config with aliases."""
    method: Optional[str] = None
    threshold: Optional[float] = None
    min_cellsize_gridpoint: Optional[int] = None
    max_cellsize_gridpoint: Optional[int] = None
    closing_kernel: Optional[tuple[int, int]] = None
    filter_by_size: Optional[bool] = None
    
    @field_validator("threshold", mode="before")
    @classmethod
    def coerce_threshold(cls, v):
        """Accept int or float for threshold."""
        if v is not None:
            return float(v)
        return v
    
    @field_validator("method", mode="before")
    @classmethod
    def normalize_method(cls, v):
        """Normalize method names to lowercase."""
        if isinstance(v, str):
            return v.lower().strip()
        return v


class UserGlobalConfig(AdaptBaseModel):
    """User-facing global config."""
    z_level: Optional[float] = None
    var_names: Optional[dict[str, str]] = None
    coord_names: Optional[dict[str, str]] = None
    
    @field_validator("z_level", mode="before")
    @classmethod
    def coerce_z_level(cls, v):
        """Accept int or float for z_level."""
        if v is not None:
            return float(v)
        return v


class UserProjectorConfig(AdaptBaseModel):
    """User-facing projector config."""
    method: Optional[str] = None
    max_time_interval_minutes: Optional[int] = None
    max_projection_steps: Optional[int] = None
    nan_fill_value: Optional[float] = None
    flow_params: Optional[dict[str, Any]] = None
    min_motion_threshold: Optional[float] = None
    
    @field_validator("method", mode="before")
    @classmethod
    def normalize_method(cls, v):
        """Normalize method names to lowercase."""
        if isinstance(v, str):
            return v.lower().strip()
        return v


class UserRegridderConfig(AdaptBaseModel):
    """User-facing regridder config."""
    grid_shape: Optional[tuple[int, int, int]] = None
    grid_limits: Optional[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = None
    roi_func: Optional[str] = None
    min_radius: Optional[float] = None
    weighting_function: Optional[str] = None
    save_netcdf: Optional[bool] = None


class UserDownloaderConfig(AdaptBaseModel):
    """User-facing downloader config."""
    radar_id: Optional[str] = None
    output_dir: Optional[str] = None
    latest_n: Optional[int] = None
    minutes: Optional[int] = None
    sleep_interval: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


class UserConfig(AdaptBaseModel):
    """User-facing configuration schema.
    
    Minimal, forgiving, and uses common aliases. Users only specify
    what they want to override from ParamConfig defaults.
    
    This config is converted to internal overrides during resolution.
    
    Usage
    -----
        user_cfg = UserConfig(
            mode="historical",
            radar_id="KHTX",
            base_dir="/data/adapt",
            z_level=2000,
            threshold_dbz=35,
        )
        
        internal = resolve_config(param_cfg, user_cfg, cli_cfg)
    """
    
    # Top-level operational settings
    mode: Optional[Literal["realtime", "historical"]] = None
    radar_id: Optional[str] = None
    base_dir: Optional[str] = None
    
    # Realtime settings
    latest_files: Optional[int] = None
    latest_minutes: Optional[int] = None
    poll_interval_sec: Optional[int] = None
    
    # Historical settings
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    
    # Grid settings (flat aliases)
    grid_shape: Optional[tuple[int, int, int]] = None
    grid_limits: Optional[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = None
    
    # Segmentation settings (flat aliases)
    z_level: Optional[float] = None
    reflectivity_var: Optional[str] = None
    segmentation_method: Optional[str] = None
    threshold_dbz: Optional[float] = None
    min_cell_size: Optional[int] = None
    max_cell_size: Optional[int] = None
    
    # Projection settings (flat aliases)
    projection_method: Optional[str] = None
    projection_steps: Optional[int] = None
    
    # Nested overrides (advanced users)
    downloader: Optional[UserDownloaderConfig] = None
    regridder: Optional[UserRegridderConfig] = None
    segmenter: Optional[UserSegmenterConfig] = None
    global_: Optional[UserGlobalConfig] = Field(None, alias="global")
    projector: Optional[UserProjectorConfig] = None
    
    model_config = AdaptBaseModel.model_config.copy()
    # Allow forgiving input dictionaries (ignore unknown legacy keys)
    model_config.update({"populate_by_name": True, "extra": "ignore"})

    @model_validator(mode="before")
    @classmethod
    def _normalize_input_keys(cls, values):
        """Normalize legacy and uppercase keys before Pydantic validation.

        This method maps common legacy UPPERCASE names and synonyms to the
        canonical snake_case field names used by `UserConfig`. It leaves
        nested dicts (e.g., `downloader`, `segmenter`) untouched so their
        own model validators handle normalization.
        Unknown keys are ignored (see model_config.extra='ignore').
        """
        if not isinstance(values, dict):
            return values

        key_map = {
            # Operational keys
            "MODE": "mode",
            "RADAR": "radar_id",
            "RADAR_ID": "radar_id",
            "BASE_DIR": "base_dir",

            # Realtime / polling
            "LATEST_FILES": "latest_files",
            "LATEST_N": "latest_files",
            "LATEST_MINUTES": "latest_minutes",
            "POLL_INTERVAL_SEC": "poll_interval_sec",
            "SLEEP_INTERVAL": "poll_interval_sec",

            # Historical
            "START_TIME": "start_time",
            "END_TIME": "end_time",

            # Grid
            "GRID_SHAPE": "grid_shape",
            "GRID_LIMITS": "grid_limits",

            # Segmenter
            "Z_LEVEL": "z_level",
            "REFLECTIVITY_VAR": "reflectivity_var",
            "SEGMENTATION_METHOD": "segmentation_method",
            "SEGMENTER_METHOD": "segmentation_method",
            "THRESHOLD_DBZ": "threshold_dbz",
            "MIN_CELL_SIZE": "min_cell_size",
            "MAX_CELL_SIZE": "max_cell_size",

            # Projection
            "PROJECTION_METHOD": "projection_method",
            "PROJECTION_STEPS": "projection_steps",
        }

        normalized: dict = {}
        for k, v in values.items():
            if not isinstance(k, str):
                # Preserve non-string keys as-is
                normalized[k] = v
                continue

            # Direct mapping for common legacy uppercase keys
            if k in key_map:
                normalized[key_map[k]] = v
                continue

            # Case-insensitive match: try uppercase form
            up = k.upper()
            if up in key_map:
                normalized[key_map[up]] = v
                continue

            # Accept already-canonical names (snake_case or mixed)
            normalized[k] = v

        return normalized
    
    @field_validator("z_level", "threshold_dbz", mode="before")
    @classmethod
    def coerce_numeric_fields(cls, v):
        """Accept int or float for numeric fields."""
        if v is not None:
            return float(v)
        return v
    
    @field_validator("segmentation_method", "projection_method", mode="before")
    @classmethod
    def normalize_method_names(cls, v):
        """Normalize method names to lowercase."""
        if isinstance(v, str):
            return v.lower().strip()
        return v
    
    def to_internal_overrides(self) -> dict:
        """Convert user config to internal config structure.
        
        Maps user-friendly flat keys to nested internal structure:
        - radar_id → downloader.radar_id
        - z_level → global.z_level
        - threshold_dbz → segmenter.threshold
        - etc.
        
        Returns
        -------
        dict
            Nested dictionary matching InternalConfig structure
        """
        overrides = {}
        
        # Top-level
        if self.mode is not None:
            overrides["mode"] = self.mode
        
        # Downloader section
        downloader = {}
        if self.radar_id is not None:
            downloader["radar_id"] = self.radar_id
        if self.start_time is not None:
            downloader["start_time"] = self.start_time
        if self.end_time is not None:
            downloader["end_time"] = self.end_time
        if self.latest_files is not None:
            downloader["latest_n"] = self.latest_files
        if self.latest_minutes is not None:
            downloader["minutes"] = self.latest_minutes
        if self.poll_interval_sec is not None:
            downloader["sleep_interval"] = self.poll_interval_sec
        
        # Map base_dir to downloader.output_dir for convenience
        if self.base_dir is not None:
            # Accept either a Path or string; keep as string for overrides
            downloader["output_dir"] = str(self.base_dir)

        # Merge with explicit downloader config
        if self.downloader is not None:
            downloader.update(self.downloader.model_dump(exclude_none=True))

        if downloader:
            overrides["downloader"] = downloader

        # Regridder section
        regridder = {}
        if self.grid_shape is not None:
            regridder["grid_shape"] = self.grid_shape
        if self.grid_limits is not None:
            regridder["grid_limits"] = self.grid_limits

        # Merge with explicit regridder config
        if self.regridder is not None:
            regridder.update(self.regridder.model_dump(exclude_none=True))
        
        if regridder:
            overrides["regridder"] = regridder
        
        # Segmenter section
        segmenter = {}
        if self.segmentation_method is not None:
            segmenter["method"] = self.segmentation_method
        if self.threshold_dbz is not None:
            segmenter["threshold"] = self.threshold_dbz
        if self.min_cell_size is not None:
            segmenter["min_cellsize_gridpoint"] = self.min_cell_size
        if self.max_cell_size is not None:
            segmenter["max_cellsize_gridpoint"] = self.max_cell_size
        
        # Merge with explicit segmenter config
        if self.segmenter is not None:
            segmenter.update(self.segmenter.model_dump(exclude_none=True))
        
        if segmenter:
            overrides["segmenter"] = segmenter
        
        # Global section
        global_cfg = {}
        if self.z_level is not None:
            global_cfg["z_level"] = self.z_level
        
        if self.reflectivity_var is not None:
            var_names = global_cfg.get("var_names", {})
            var_names["reflectivity"] = self.reflectivity_var
            global_cfg["var_names"] = var_names
        
        # Merge with explicit global config
        if self.global_ is not None:
            global_cfg.update(self.global_.model_dump(exclude_none=True))
        
        if global_cfg:
            overrides["global"] = global_cfg
        
        # Projector section
        projector = {}
        if self.projection_method is not None:
            projector["method"] = self.projection_method
        if self.projection_steps is not None:
            projector["max_projection_steps"] = self.projection_steps
        
        # Merge with explicit projector config
        if self.projector is not None:
            projector.update(self.projector.model_dump(exclude_none=True))
        
        if projector:
            overrides["projector"] = projector
        
        return overrides
