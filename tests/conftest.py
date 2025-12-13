# tests/conftest.py
from datetime import datetime, timezone
import pytest


class FakeScan:
    def __init__(self, key, scan_time=None):
        self.key = key
        self.scan_time = scan_time or datetime.now(timezone.utc)


class FakeAwsConn:
    def __init__(self, scans):
        self.scans = scans

    def get_avail_scans_in_range(self, start, end, radar_id):
        return self.scans

    def download(self, scans, target_dir, keep_aws_folders=False):
        class Result:
            def __init__(self, path):
                self.filepath = path

        results = []
        for scan in scans:
            path = target_dir / scan.key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * 2048)
            results.append(Result(path))

        class DownloadResults:
            def iter_success(self):
                return results

        return DownloadResults()


@pytest.fixture
def fake_scan():
    return FakeScan


@pytest.fixture
def fake_aws_conn():
    return FakeAwsConn
