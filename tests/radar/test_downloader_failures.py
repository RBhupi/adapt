# tests/test_downloader_failures.py
import pytest
from adapt.radar.downloader import AwsNexradDownloader
from datetime import datetime, timezone

pytestmark = pytest.mark.unit


def test_download_failure_does_not_queue(tmp_path, fake_scan):
    class FailingConn:
        def get_avail_scans_in_range(self, *a):
            return [fake_scan("bad", datetime.now(timezone.utc))]

        def download(self, *a, **k):
            class R:
                def iter_success(self): return []
            return R()

    d = AwsNexradDownloader(
        {"output_dir": tmp_path},
        conn=FailingConn()
    )

    downloads = d._download_realtime()
    assert downloads == []


def test_fetch_scans_exception_returns_empty(tmp_path):
    class ExplodingConn:
        def get_avail_scans_in_range(self, *a):
            raise RuntimeError("AWS down")

    d = AwsNexradDownloader(
        {"output_dir": tmp_path},
        conn=ExplodingConn()
    )

    scans = d._fetch_scans(datetime.now(timezone.utc), datetime.now(timezone.utc))
    assert scans == []
