import pytest
from adapt.radar.downloader import AwsNexradDownloader
from adapt.schemas.user import UserConfig
from datetime import datetime, timedelta, timezone


@pytest.mark.integration
def test_real_aws_listing(tmp_path, make_config):
    """Test real AWS NEXRAD data listing.
    
    Uses a known radar ID (KMOB) to ensure we get real data from AWS.
    Skips if no scans are available (expected during low-activity periods).
    """
    config = make_config(radar_id="KHTX")  # Use a known radar with consistent data
    d = AwsNexradDownloader(config, output_dir=tmp_path)
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=60)
    scans = d._fetch_scans(start, end)
    
    # Skip if no scans available (integration test depends on real AWS data availability)
    if not scans:
        pytest.skip("No NEXRAD scans available in AWS for the past 60 minutes (expected during low-activity periods)")

