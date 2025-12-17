# tests/test_downloader_queue.py
from queue import Queue
from datetime import datetime, timezone

from adapt.radar.downloader import AwsNexradDownloader

import pytest

pytestmark = pytest.mark.unit


def test_notify_queue_puts_item(tmp_path, make_config):
    q = Queue()
    config = make_config()
    d = AwsNexradDownloader(config, output_dir=tmp_path, result_queue=q)

    path = tmp_path / "file1"

    d._notify_queue(
        path=path,
        scan_time=datetime.now(timezone.utc),
        is_new=True,
    )

    item = q.get_nowait()

    assert item["radar_id"] == d.radar_id
    assert item["path"] == path
    assert "scan_time" in item
    assert "file_id" in item


def test_notify_queue_calls_tracker(tmp_path, fake_scan, make_config):
    class FakeTracker:
        def __init__(self):
            self.registered = False

        def register_file(self, *a, **k):
            self.registered = True

        def mark_stage_complete(self, *a, **k):
            pass

    tracker = FakeTracker()
    from queue import Queue

    q = Queue()
    config = make_config()
    d = AwsNexradDownloader(config, output_dir=tmp_path, result_queue=q, file_tracker=tracker)

    d._notify_queue(
        path=tmp_path / "f", scan_time=fake_scan("x").scan_time, is_new=True
    )

    assert tracker.registered
