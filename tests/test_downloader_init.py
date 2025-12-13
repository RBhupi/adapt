# tests/test_downloader_init.py
from adapt.radar.downloader import AwsNexradDownloader


def test_init_custom_config(tmp_path):
    config = {
        "radar_id": "KDIX",
        "output_dir": tmp_path,
        "sleep_interval": 10,
        "latest_n": 5,
        "minutes": 60,
    }

    d = AwsNexradDownloader(config)

    assert d.radar_id == "KDIX"
    assert d.output_dir == tmp_path
    assert d.sleep_interval == 10
    assert d.latest_n == 5
    assert d.minutes == 60


def test_stop_sets_event():
    d = AwsNexradDownloader({})
    assert not d.stopped()
    d.stop()
    assert d.stopped()


def test_is_historical_mode():
    d = AwsNexradDownloader({
        "start_time": "2024-01-01T00:00:00Z",
        "end_time": "2024-01-01T01:00:00Z",
    })
    assert d.is_historical_mode() is True

    d2 = AwsNexradDownloader({})
    assert d2.is_historical_mode() is False


def test_parse_time_range():
    d = AwsNexradDownloader({
        "start_time": "2024-01-01T00:00:00Z",
        "end_time": "2024-01-01T01:00:00Z",
    })

    start, end = d._parse_time_range()

    assert start.tzinfo is not None
    assert end > start
    assert (end - start).total_seconds() == 3600

def test_file_exists_rejects_small_files(tmp_path):
    d = AwsNexradDownloader({"output_dir": tmp_path})

    p = tmp_path / "tiny"
    p.write_bytes(b"x")

    assert not d._file_exists(p)
