"""Tests for CLIConfig schema and conversion to internal overrides."""

import pytest
from adapt.schemas.cli import CLIConfig


def test_cli_to_internal_overrides_with_mode():
    """Test CLI config conversion with mode override."""
    cli = CLIConfig(mode="historical")
    overrides = cli.to_internal_overrides()
    assert overrides["mode"] == "historical"


def test_cli_to_internal_overrides_with_realtime_mode():
    """Test CLI config conversion with realtime mode."""
    cli = CLIConfig(mode="realtime")
    overrides = cli.to_internal_overrides()
    assert overrides["mode"] == "realtime"


def test_cli_to_internal_overrides_with_radar_id():
    """Test CLI config conversion with radar_id override."""
    cli = CLIConfig(radar_id="KMOB")
    overrides = cli.to_internal_overrides()
    assert overrides["downloader"]["radar_id"] == "KMOB"


def test_cli_to_internal_overrides_with_log_level():
    """Test CLI config conversion with log_level override."""
    cli = CLIConfig(log_level="DEBUG")
    overrides = cli.to_internal_overrides()
    assert overrides["logging"]["level"] == "DEBUG"


def test_cli_to_internal_overrides_with_multiple_fields():
    """Test CLI config conversion with multiple overrides."""
    cli = CLIConfig(mode="historical", radar_id="KHTX", log_level="INFO")
    overrides = cli.to_internal_overrides()
    assert overrides["mode"] == "historical"
    assert overrides["downloader"]["radar_id"] == "KHTX"
    assert overrides["logging"]["level"] == "INFO"


def test_cli_to_internal_overrides_empty():
    """Test CLI config conversion with no overrides."""
    cli = CLIConfig()
    overrides = cli.to_internal_overrides()
    assert overrides == {}


def test_cli_config_accepts_base_dir():
    """Test that base_dir is accepted but not in overrides."""
    cli = CLIConfig(base_dir="/path/to/output")
    assert cli.base_dir == "/path/to/output"
    overrides = cli.to_internal_overrides()
    # base_dir handled separately by setup_output_directories
    assert "base_dir" not in overrides


def test_cli_config_all_log_levels():
    """Test all valid log levels."""
    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        cli = CLIConfig(log_level=level)
        overrides = cli.to_internal_overrides()
        assert overrides["logging"]["level"] == level
