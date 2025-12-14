# tests/test_downloader_historical.py
from datetime import datetime, timezone
import pytest
from adapt.radar.downloader import AwsNexradDownloader

pytestmark = pytest.mark.unit


def test_historical_mode_completes(tmp_path, fake_scan, fake_aws_conn):
    scans = [
        fake_scan("h1", datetime(2024, 1, 1, tzinfo=timezone.utc)),
        fake_scan("h2", datetime(2024, 1, 1, 1, tzinfo=timezone.utc)),
    ]

    d = AwsNexradDownloader(
        config={
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-01T02:00:00Z",
            "output_dir": tmp_path,
        },
        conn=fake_aws_conn(scans),
        sleeper=lambda _: None,
    )

    downloads = d.download_task()

    assert d.is_historical_complete()
    processed, expected = d.get_historical_progress()
    assert expected == 2
    assert len(downloads) == 2

# Mock AWS completely.
def test_fetch_scans_filters_and_sorts(monkeypatch):
    class FakeScan:
        def __init__(self, key, scan_time):
            self.key = key
            self.scan_time = scan_time

    scans = [
        FakeScan("file2", datetime(2024,1,1,1)),
        FakeScan("file1_MDM", datetime(2024,1,1,0)),
        FakeScan("file0", datetime(2024,1,1,0)),
    ]

    d = AwsNexradDownloader({})
    monkeypatch.setattr(
        d.conn,
        "get_avail_scans_in_range",
        lambda *args, **kwargs: scans
    )

    result = d._fetch_scans(datetime.now(), datetime.now())
    assert len(result) == 2
    assert result[0].key == "file0"
