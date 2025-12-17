# tests/test_downloader_realtime.py
from queue import Queue
from datetime import datetime, timezone

from adapt.radar.downloader import AwsNexradDownloader


import pytest

pytestmark = pytest.mark.unit


def test_realtime_download_hybrid(tmp_path, fake_scan, fake_aws_conn, make_config):
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)

    scans = [
        fake_scan("scan1", now),
        fake_scan("scan2", now),
    ]

    q = Queue()

    config = make_config(
        radar_id="KDIX",
        latest_files=2,
        latest_minutes=30,
    )

    d = AwsNexradDownloader(
        config,
        output_dir=tmp_path,
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


def test_realtime_idempotent(tmp_path, fake_scan, fake_aws_conn, make_config):
    scans = [fake_scan("same")]

    config = make_config()
    d = AwsNexradDownloader(
        config,
        output_dir=tmp_path,
        conn=fake_aws_conn(scans),
        sleeper=lambda _: None,
    )

    d._download_realtime()
    d._download_realtime()

    assert len(d._known_files) == 1
