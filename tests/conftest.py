"""Root-level pytest fixtures for ADAPT test suite.

Provides shared configuration fixtures following Pydantic-based architecture.
All tests must use these fixtures instead of creating raw dict configs.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from adapt.schemas import ParamConfig, UserConfig, CLIConfig, InternalConfig, resolve_config


# =============================================================================
# Configuration Fixtures (Pydantic-based)
# =============================================================================

@pytest.fixture
def param_config():
    """Expert configuration with all defaults.
    
    Use this as the base for all test configs. Override specific values
    using user_config or by creating custom UserConfig instances.
    """
    return ParamConfig()


@pytest.fixture
def internal_config(param_config):
    """Fully validated runtime configuration (no overrides).
    
    Use this when tests don't care about specific config values and just
    need a valid InternalConfig to pass to constructors.
    
    Examples
    --------
    >>> def test_segmenter_init(internal_config):
    ...     seg = RadarCellSegmenter(internal_config)
    ...     assert seg.method == "threshold"
    """
    return resolve_config(param_config, None, None)


@pytest.fixture
def make_config(param_config):
    """Factory fixture for creating custom test configs.
    
    Use this when you need to override specific values for a test.
    Returns a callable that accepts UserConfig-compatible kwargs.
    
    Examples
    --------
    >>> def test_custom_threshold(make_config):
    ...     config = make_config(threshold_dbz=35)
    ...     seg = RadarCellSegmenter(config)
    ...     assert seg.threshold == 35.0
    """
    def _make(**user_overrides):
        """Create InternalConfig with user overrides."""
        if user_overrides:
            user = UserConfig(**user_overrides)
            return resolve_config(param_config, user, None)
        else:
            return resolve_config(param_config, None, None)
    
    return _make


# =============================================================================
# Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Temporary directory that is cleaned up after test."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def output_dirs(temp_dir):
    """Standard ADAPT output directory structure.
    
    Returns dict with keys: nexrad, gridnc, analysis, plots, logs
    All directories are created and cleaned up automatically.
    """
    dirs = {
        "nexrad": temp_dir / "nexrad",
        "gridded": temp_dir / "gridded",
        "gridnc": temp_dir / "gridnc",  # Alias for gridded
        "analysis": temp_dir / "analysis",
        "plots": temp_dir / "plots",
        "logs": temp_dir / "logs",
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    return dirs
