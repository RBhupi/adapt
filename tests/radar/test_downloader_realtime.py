# tests/test_downloader_realtime.py
from queue import Queue
from datetime import datetime, timezone

from adapt.radar.downloader import AwsNexradDownloader


def test_realtime_download_hybrid(tmp_path, fake_scan, fake_aws_conn):
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)

    scans = [
        fake_scan("scan1", now),
        fake_scan("scan2", now),
    ]

    q = Queue()

    d = AwsNexradDownloader(
        config={
            "radar_id": "KDIX",
            "output_dir": tmp_path,
            "latest_n": 2,
            "minutes": 30,
        },
        result_queue=q,
        conn=fake_aws_conn(scans),
        clock=lambda: now,
        sleeper=lambda _: None,
    )

    downloads = d._download_realtime()

    assert len(downloads) == 2
    assert q.qsize() == 2

    for path in downloads:
        assert path.exists()
        assert path.stat().st_size >= 1024


def test_realtime_idempotent(tmp_path, fake_scan, fake_aws_conn):
    scans = [fake_scan("same")]

    d = AwsNexradDownloader(
        {"output_dir": tmp_path},
        conn=fake_aws_conn(scans),
        sleeper=lambda _: None,
    )

    d._download_realtime()
    d._download_realtime()

    assert len(d._known_files) == 1
