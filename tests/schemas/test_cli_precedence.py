from adapt.schemas.user import UserConfig
from adapt.schemas.cli import CLIConfig
from adapt.schemas.param import ParamConfig
from adapt.schemas.resolve import resolve_config


def test_cli_overrides_do_not_mutate_user():
    user = UserConfig.model_validate({"RADAR_ID": "KABC", "MODE": "realtime", "BASE_DIR": "/tmp"})

    cli = CLIConfig.model_validate({"radar_id": "KHTX"})

    internal = resolve_config(ParamConfig(), user, cli)

    # CLI should take precedence
    assert internal.downloader.radar_id == "KHTX"

    # But the original user model should remain unchanged
    assert user.radar_id == "KABC"


def test_cli_minimal_overrides_radar_id():
    """CLI radar_id override should work correctly."""
    user = UserConfig(base_dir="/tmp", radar_id="KABC")
    cli = CLIConfig(radar_id="KHTX")
    
    config = resolve_config(ParamConfig(), user, cli)
    
    assert config.downloader.radar_id == "KHTX"  # CLI wins
    assert config.base_dir == "/tmp"  # User value preserved


def test_cli_minimal_overrides_mode():
    """CLI mode override should work correctly.""" 
    user = UserConfig(
        base_dir="/tmp", 
        radar_id="KABC", 
        mode="realtime",
        start_time="2024-01-01T00:00:00Z",
        end_time="2024-01-01T12:00:00Z"
    )
    cli = CLIConfig(mode="historical")
    
    config = resolve_config(ParamConfig(), user, cli)
    
    assert config.mode == "historical"  # CLI wins
    assert config.downloader.radar_id == "KABC"  # User value preserved
    # Historical mode validation should pass since start/end times provided


def test_cli_precedence_no_user_config():
    """CLI should work even without UserConfig."""
    cli = CLIConfig(radar_id="KHTX", mode="realtime")
    
    # This will need minimal UserConfig for required fields
    user = UserConfig(base_dir="/tmp")
    config = resolve_config(ParamConfig(), user, cli)
    
    assert config.downloader.radar_id == "KHTX"
    assert config.mode == "realtime"


def test_cli_only_overrides_specified_fields():
    """CLI should only override fields that are explicitly set."""
    user = UserConfig(
        base_dir="/tmp",
        radar_id="KABC", 
        mode="realtime",
        threshold=35
    )
    
    # CLI only sets radar_id
    cli = CLIConfig(radar_id="KHTX")
    
    config = resolve_config(ParamConfig(), user, cli)
    
    assert config.downloader.radar_id == "KHTX"  # CLI override
    assert config.mode == "realtime"  # User value preserved
    assert config.segmenter.threshold == 35.0  # User value preserved
