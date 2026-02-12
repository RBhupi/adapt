# tests/test_downloader_init.py
import pytest
from adapt.radar.downloader import AwsNexradDownloader

pytestmark = pytest.mark.unit


def test_init_custom_config(make_config, radar_output_dirs):
    """Downloader initializes with custom config."""
    from adapt.schemas.user import UserDownloaderConfig
    config = make_config(downloader=UserDownloaderConfig(radar_id="KDIX", latest_files=5, latest_minutes=60))
    d = AwsNexradDownloader(config, radar_output_dirs["nexrad"])

    assert d.config.downloader.radar_id == "KDIX"
    assert d.config.downloader.latest_files == 5
    assert d.config.downloader.latest_minutes == 60


def test_stop_sets_event(radar_config, radar_output_dirs):
    """Stop event prevents downloader from polling."""
    d = AwsNexradDownloader(radar_config, radar_output_dirs["nexrad"])
    assert not d.stopped()
    d.stop()
    assert d.stopped()


def test_historical_mode_from_config(make_config, radar_output_dirs):
    """Downloader detects historical mode from config.downloader.mode."""
    from adapt.schemas.user import UserDownloaderConfig
    config = make_config(
        downloader=UserDownloaderConfig(
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T01:00:00Z",
        )
    )
    d = AwsNexradDownloader(config, radar_output_dirs["nexrad"])
    # Mode is decided by schema, not by is_historical_mode() method
    assert d.config.downloader.mode == "historical"

    d2 = AwsNexradDownloader(make_config(), radar_output_dirs["nexrad"])
    assert d2.config.downloader.mode == "realtime"


def test_parse_time_range(make_config, radar_output_dirs):
    """Downloader parses time range correctly."""
    from adapt.schemas.user import UserDownloaderConfig
    config = make_config(
        downloader=UserDownloaderConfig(
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T01:00:00Z",
        )
    )
    d = AwsNexradDownloader(config, radar_output_dirs["nexrad"])

    start, end = d._parse_time_range()

    assert start.tzinfo is not None
    assert end > start
    assert (end - start).total_seconds() == 3600


def test_file_exists_rejects_small_files(tmp_path, radar_config, radar_output_dirs):
    """Downloader rejects files below minimum size."""
    d = AwsNexradDownloader(radar_config, tmp_path)

    p = tmp_path / "tiny"
    p.write_bytes(b"x")

    assert not d._file_exists(p)


def test_file_exists_true(tmp_path, radar_config, radar_output_dirs):
    """Downloader accepts files above minimum size."""
    d = AwsNexradDownloader(radar_config, radar_output_dirs["nexrad"])
    p = tmp_path / "f"
    p.write_bytes(b"x" * 2048)
    assert d._file_exists(p)


from datetime import datetime
def test_get_local_path(make_config, radar_output_dirs):
    """Downloader generates correct local file paths with new structure."""
    class FakeScan:
        key = "foo/bar/testfile"
        scan_time = datetime(2024, 1, 1)

    from adapt.schemas.user import UserDownloaderConfig
    config = make_config(downloader=UserDownloaderConfig(radar_id="KDIX"))
    # Use output_dirs for new path structure (RADAR_ID/nexrad/YYYYMMDD/)
    d = AwsNexradDownloader(config, output_dirs=radar_output_dirs)

    path = d._get_local_path(FakeScan())
    assert "20240101" in str(path)
    assert "KDIX" in str(path)
    assert path.name == "testfile"
    # New structure: base/KDIX/nexrad/20240101/testfile
    assert "KDIX/nexrad/20240101" in str(path) or "KDIX\\nexrad\\20240101" in str(path)


def test_get_local_path_legacy(make_config, radar_output_dirs):
    """Downloader generates correct local file paths with legacy output_dir."""
    class FakeScan:
        key = "foo/bar/testfile"
        scan_time = datetime(2024, 1, 1)

    from adapt.schemas.user import UserDownloaderConfig
    config = make_config(downloader=UserDownloaderConfig(radar_id="KDIX"))
    # Use legacy output_dir parameter
    d = AwsNexradDownloader(config, output_dir=radar_output_dirs["nexrad"])

    path = d._get_local_path(FakeScan())
    assert "20240101" in str(path)
    assert "KDIX" in str(path)
    assert path.name == "testfile"
