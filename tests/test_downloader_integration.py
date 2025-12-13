import pytest
from adapt.radar.downloader import AwsNexradDownloader
from datetime import datetime, timedelta, timezone


@pytest.mark.integration
def test_real_aws_listing():
    d = AwsNexradDownloader({"radar_id": "KTLX"})
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=60)
    scans = d._fetch_scans(start, end)
    assert len(scans) > 0

