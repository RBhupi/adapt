import pytest

from adapt.schemas.user import UserConfig


def test_uppercase_keys_are_handled():
    raw = {
        "MODE": "historical",
        "RADAR_ID": "KHTX",
        "THRESHOLD_DBZ": 40,
        "BASE_DIR": "/tmp/adapt_out",
    }

    user = UserConfig.model_validate(raw)

    assert user.mode == "historical"
    assert user.radar_id == "KHTX"
    assert isinstance(user.threshold, float) and user.threshold == 40.0
    assert user.base_dir == "/tmp/adapt_out"


def test_unknown_keys_are_ignored():
    raw = {"MODE": "realtime", "UNKNOWN_LEGACY": 12345}
    user = UserConfig.model_validate(raw)

    assert user.mode == "realtime"
    # Unknown key should not become an attribute nor raise
    assert not hasattr(user, "UNKNOWN_LEGACY")
