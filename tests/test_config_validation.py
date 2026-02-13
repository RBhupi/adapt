"""Tests for conditional configuration validation.

Tests that:
- Historical mode REQUIRES start_time and end_time
- Realtime mode allows None start_time and end_time
- Mode is automatically inferred from start_time/end_time
"""

import pytest
from pydantic import ValidationError
from adapt.schemas.param import ParamConfig
from adapt.schemas.user import UserConfig
from adapt.schemas.cli import CLIConfig
from adapt.schemas.resolve import resolve_config


class TestHistoricalModeValidation:
    """Conditional validation for historical mode."""
    
    def test_historical_mode_requires_start_time(self):
        """Historical mode must have start_time or raise ValueError."""
        param = ParamConfig()
        user = UserConfig(
            mode="historical",
            radar_id="KHTX",
            base_dir="/tmp/test",
            end_time="2025-12-01T23:59:59Z",
            # MISSING: start_time
        )
        
        with pytest.raises(ValueError) as exc_info:
            resolve_config(param, user)
        
        assert "start_time" in str(exc_info.value).lower()
    
    def test_historical_mode_requires_end_time(self):
        """Historical mode must have end_time or raise ValueError."""
        param = ParamConfig()
        user = UserConfig(
            mode="historical",
            radar_id="KHTX",
            base_dir="/tmp/test",
            start_time="2025-12-01T00:00:00Z",
            # MISSING: end_time
        )
        
        with pytest.raises(ValueError) as exc_info:
            resolve_config(param, user)
        
        assert "end_time" in str(exc_info.value).lower()
    
    def test_historical_mode_inference_from_user_config(self):
        """Mode is automatically set to historical if times provided in UserConfig."""
        param = ParamConfig()
        user = UserConfig(
            radar_id="KHTX",
            base_dir="/tmp/test",
            start_time="2025-12-01T00:00:00Z",
            end_time="2025-12-01T23:59:59Z",
        )
        
        config = resolve_config(param, user)
        assert config.mode == "historical"
        assert config.downloader.mode == "historical"


class TestRealtimeModeValidation:
    """Validation for realtime mode (times are optional)."""
    
    def test_realtime_mode_allows_missing_times(self):
        """Realtime mode works without start_time and end_time."""
        param = ParamConfig()
        user = UserConfig(
            mode="realtime",
            radar_id="KHTX",
            base_dir="/tmp/test",
        )
        
        config = resolve_config(param, user)
        assert config.mode == "realtime"
        assert config.downloader.start_time is None
    
    def test_realtime_mode_allows_explicit_times(self):
        """Realtime mode can optionally have times without switching to historical."""
        param = ParamConfig()
        user = UserConfig(
            mode="realtime",
            radar_id="KHTX",
            base_dir="/tmp/test",
            start_time="2025-12-01T00:00:00Z",
            end_time="2025-12-01T23:59:59Z",
        )
        
        config = resolve_config(param, user)
        assert config.mode == "realtime"
        assert config.downloader.mode == "realtime"


class TestCLIOverrideValidation:
    """CLI overrides respect conditional validation and inference."""
    
    def test_cli_historical_inference(self):
        """CLI override with times automatically sets mode to historical."""
        param = ParamConfig()
        user = UserConfig(radar_id="KHTX", base_dir="/tmp/test")
        cli = CLIConfig(
            start_time="2025-12-01T00:00:00Z",
            end_time="2025-12-01T23:59:59Z"
        )
        
        config = resolve_config(param, user, cli)
        assert config.mode == "historical"
        assert config.downloader.mode == "historical"
