# tests/test_downloader_run.py
from adapt.radar.downloader import AwsNexradDownloader


def test_run_exits_after_historical_complete(tmp_path, fake_scan, fake_aws_conn):
    scans = [fake_scan("one")]

    d = AwsNexradDownloader(
        {
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-01T01:00:00Z",
            "output_dir": tmp_path,
        },
        conn=fake_aws_conn(scans),
        sleeper=lambda _: None,
    )

    d.run()

    assert d.is_historical_complete()
