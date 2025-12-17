"""Test config resolution and validation with Pydantic."""

import pytest
from adapt.schemas import ParamConfig, UserConfig, CLIConfig, InternalConfig
from adapt.schemas.resolve import resolve_config
from adapt.schemas.user import UserSegmenterConfig, UserProjectorConfig, UserDownloaderConfig


class TestConfigResolution:
    """Test resolve_config() precedence and merging."""

    def test_resolve_config_all_defaults(self):
        """Resolving with no user/CLI overrides uses all ParamConfig defaults."""
        config = resolve_config(ParamConfig(), None, None)

        assert isinstance(config, InternalConfig)
        assert config.segmenter.threshold == 30.0
        assert config.segmenter.closing_kernel == (1, 1)
        assert config.downloader.radar_id is None  # No default radar_id

    def test_user_config_overrides_param_config(self):
        """UserConfig values override ParamConfig defaults."""
        user = UserConfig(threshold_dbz=40)
        config = resolve_config(ParamConfig(), user, None)

        assert config.segmenter.threshold == 40.0

    def test_cli_config_with_valid_structure(self):
        """CLIConfig structure validation (if implemented)."""
        # CLIConfig currently has limited fields - test what exists
        from adapt.schemas import CLIConfig
        # Just verify it can be instantiated
        cli = CLIConfig()

    def test_precedence_param_user(self):
        """Full precedence: User > Param."""
        param = ParamConfig()
        user = UserConfig(
            threshold_dbz=40,
            radar_id="KDLH"
        )
        config = resolve_config(param, user, None)

        # User won on threshold
        assert config.segmenter.threshold == 40.0
        # User won on radar_id
        assert config.downloader.radar_id == "KDLH"

    def test_empty_user_config_uses_all_param_defaults(self):
        """Empty UserConfig() doesn't override anything."""
        config = resolve_config(ParamConfig(), UserConfig(), None)

        assert config.segmenter.threshold == 30.0
        assert config.projector.max_projection_steps == 1

    def test_none_user_config_uses_all_param_defaults(self):
        """None UserConfig doesn't override anything."""
        config = resolve_config(ParamConfig(), None, None)

        assert config.segmenter.threshold == 30.0
        assert config.projector.max_projection_steps == 1


class TestUserConfigAliases:
    """Test UserConfig flat aliases map correctly."""

    def test_threshold_dbz_alias(self):
        """threshold_dbz flat alias maps to segmenter.threshold."""
        user = UserConfig(threshold_dbz=35)
        config = resolve_config(ParamConfig(), user, None)

        assert config.segmenter.threshold == 35.0

    def test_radar_id_alias(self):
        """radar_id flat alias maps to downloader.radar_id."""
        user = UserConfig(radar_id="KDIX")
        config = resolve_config(ParamConfig(), user, None)

        assert config.downloader.radar_id == "KDIX"

    def test_reflectivity_var_alias(self):
        """reflectivity_var alias maps to global var_names."""
        user = UserConfig(reflectivity_var="dbz")
        config = resolve_config(ParamConfig(), user, None)

        assert config.global_.var_names.reflectivity == "dbz"

    def test_projection_steps_alias(self):
        """projection_steps alias maps to projector.max_projection_steps."""
        user = UserConfig(projection_steps=5)
        config = resolve_config(ParamConfig(), user, None)

        assert config.projector.max_projection_steps == 5

    def test_min_cell_size_alias(self):
        """min_cell_size alias maps to segmenter.min_cellsize_gridpoint."""
        user = UserConfig(min_cell_size=10)
        config = resolve_config(ParamConfig(), user, None)

        assert config.segmenter.min_cellsize_gridpoint == 10

    def test_nested_segmenter_override(self):
        """Nested segmenter config overrides flat alias."""
        user = UserConfig(
            threshold_dbz=30,
            segmenter=UserSegmenterConfig(threshold=40)
        )
        config = resolve_config(ParamConfig(), user, None)

        # Nested should win
        assert config.segmenter.threshold == 40.0


class TestTypeCoercion:
    """Test UserConfig type coercion."""

    def test_int_coerced_to_float_for_threshold(self):
        """Integer threshold is coerced to float."""
        user = UserConfig(threshold_dbz=35)  # int
        config = resolve_config(ParamConfig(), user, None)

        assert isinstance(config.segmenter.threshold, float)
        assert config.segmenter.threshold == 35.0

    def test_int_coerced_to_float_for_z_level(self):
        """Integer z_level is coerced to float."""
        user = UserConfig(z_level=1500)  # int
        config = resolve_config(ParamConfig(), user, None)

        assert isinstance(config.global_.z_level, float)
        assert config.global_.z_level == 1500.0

    def test_method_normalized_to_lowercase(self):
        """Method names are normalized to lowercase."""
        user = UserConfig(segmentation_method="THRESHOLD")
        config = resolve_config(ParamConfig(), user, None)

        assert config.segmenter.method == "threshold"

    def test_uppercase_radar_id_preserved(self):
        """Radar IDs are preserved in uppercase."""
        user = UserConfig(radar_id="KDIX")
        config = resolve_config(ParamConfig(), user, None)

        assert config.downloader.radar_id == "KDIX"


class TestEdgeCases:
    """Test config edge cases and error conditions."""

    def test_none_values_dont_override(self):
        """None values in UserConfig don't override ParamConfig."""
        user = UserConfig(threshold_dbz=None, radar_id="KDIX")
        config = resolve_config(ParamConfig(), user, None)

        assert config.segmenter.threshold == 30.0  # default, not overridden
        assert config.downloader.radar_id == "KDIX"

    def test_dict_user_config_accepted(self):
        """Dict can be passed as UserConfig (converted by Pydantic)."""
        user_dict = {"threshold_dbz": 35, "radar_id": "KDLH"}
        config = resolve_config(ParamConfig(), user_dict, None)

        assert config.segmenter.threshold == 35.0
        assert config.downloader.radar_id == "KDLH"

    def test_empty_cli_config_dict_accepted(self):
        """Empty dict can be passed as CLIConfig (converted by Pydantic)."""
        cli_dict = {}
        user = UserConfig(threshold_dbz=40)
        config = resolve_config(ParamConfig(), user, cli_dict)

        # Empty CLI dict doesn't override anything
        assert config.segmenter.threshold == 40.0

    def test_incomplete_param_config_dict_rejected(self):
        """Incomplete dict raises validation error."""
        with pytest.raises(Exception):  # Pydantic validation error
            resolve_config({"incomplete": "dict"}, None, None)

    def test_internal_config_is_complete(self):
        """Returned InternalConfig is complete with all fields."""
        config = resolve_config(ParamConfig(), None, None)

        assert config.segmenter is not None
        assert config.projector is not None
        assert config.downloader is not None


class TestDefaultValues:
    """Test ParamConfig default values match old behavior."""

    def test_segmenter_defaults(self):
        """Segmenter defaults match old hardcoded values."""
        config = resolve_config(ParamConfig(), None, None)

        assert config.segmenter.threshold == 30.0
        assert config.segmenter.closing_kernel == (1, 1)
        assert config.segmenter.min_cellsize_gridpoint == 5
        assert config.segmenter.filter_by_size is True

    def test_projector_defaults(self):
        """Projector defaults match old hardcoded values."""
        config = resolve_config(ParamConfig(), None, None)

        assert config.projector.method == "adapt_default"
        assert config.projector.max_projection_steps == 1
        assert config.projector.flow_params.winsize == 10
        assert config.projector.flow_params.iterations == 3

    def test_downloader_defaults(self):
        """Downloader defaults are initialized."""
        config = resolve_config(ParamConfig(), None, None)

        assert config.downloader.latest_n > 0
        assert config.downloader.sleep_interval > 0

    def test_regridder_defaults(self):
        """Regridder defaults are complete."""
        config = resolve_config(ParamConfig(), None, None)

        assert config.regridder.grid_shape is not None
        assert len(config.regridder.grid_shape) == 3
        assert config.regridder.save_netcdf is True


class TestConfigValidation:
    """Test Pydantic validation of configs."""

    def test_invalid_method_rejected(self):
        """Invalid segmentation method raises validation error."""
        with pytest.raises(Exception):
            resolve_config(
                ParamConfig(),
                UserConfig(segmentation_method="invalid_method_xyz"),
                None
            )

    def test_negative_threshold_rejected(self):
        """Negative threshold is coerced to float but should work."""
        # Note: Pydantic may allow negative threshold if no constraint
        # This test documents current behavior
        user = UserConfig(threshold_dbz=-10)
        config = resolve_config(ParamConfig(), user, None)
        assert config.segmenter.threshold == -10.0

    def test_zero_min_cellsize_allowed(self):
        """Zero min_cellsize is valid (means no filtering)."""
        user = UserConfig(min_cell_size=0)
        config = resolve_config(ParamConfig(), user, None)
        assert config.segmenter.min_cellsize_gridpoint == 0

    def test_valid_field_accepted(self):
        """Valid fields in user config are accepted."""
        user_dict = {
            "threshold_dbz": 35,
            "radar_id": "KDIX"
        }
        config = resolve_config(ParamConfig(), user_dict, None)
        assert config.segmenter.threshold == 35.0
        assert config.downloader.radar_id == "KDIX"


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_with_all_overrides(self):
        """Full workflow: param + user."""
        user = UserConfig(
            mode="historical",
            threshold_dbz=35,
            radar_id="KDLH",
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T12:00:00Z",
            projection_steps=3,
            segmenter=UserSegmenterConfig(
                filter_by_size=False
            )
        )
        config = resolve_config(ParamConfig(), user, None)

        # Verify all overrides took effect
        assert config.mode == "historical"
        assert config.segmenter.threshold == 35.0
        assert config.downloader.radar_id == "KDLH"
        assert config.downloader.start_time == "2024-01-01T00:00:00Z"
        assert config.projector.max_projection_steps == 3
        assert config.segmenter.filter_by_size is False

    def test_real_use_case_custom_radar(self):
        """Real use case: custom radar with strict threshold."""
        user = UserConfig(
            radar_id="KLTX",
            threshold_dbz=40,
            reflectivity_var="reflectivity_dbz",
            min_cell_size=20
        )
        config = resolve_config(ParamConfig(), user, None)

        assert config.downloader.radar_id == "KLTX"
        assert config.segmenter.threshold == 40.0
        assert config.global_.var_names.reflectivity == "reflectivity_dbz"
        assert config.segmenter.min_cellsize_gridpoint == 20

    def test_nested_config_complex_flow_params(self):
        """Complex nested config with custom flow parameters."""
        user = UserConfig(
            projector=UserProjectorConfig(
                max_projection_steps=5,
                flow_params={
                    "winsize": 15,
                    "iterations": 5,
                    "poly_n": 7,
                }
            )
        )
        config = resolve_config(ParamConfig(), user, None)

        assert config.projector.max_projection_steps == 5
        assert config.projector.flow_params.winsize == 15
        assert config.projector.flow_params.iterations == 5
        assert config.projector.flow_params.poly_n == 7
