from adapt.schemas import UserConfig, CLIConfig, ParamConfig, resolve_config


def test_cli_overrides_do_not_mutate_user():
    user = UserConfig.model_validate({"RADAR_ID": "KABC", "MODE": "realtime", "BASE_DIR": "/tmp"})

    cli = CLIConfig.model_validate({"radar_id": "KHTX"})

    internal = resolve_config(ParamConfig(), user, cli)

    # CLI should take precedence
    assert internal.downloader.radar_id == "KHTX"

    # But the original user model should remain unchanged
    assert user.radar_id == "KABC"
