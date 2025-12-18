"""Test radar availability warning behavior.

Verifies that availability warnings are only logged when the check explicitly
succeeds and finds no radar (not when the check fails/times out).
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
import logging

from adapt.radar.downloader import AwsNexradDownloader

pytestmark = pytest.mark.unit


def test_availability_check_warns_when_radar_explicitly_not_found(caplog, temp_dir):
    """Warn when availability check succeeds but radar not found."""
    config = MagicMock()
    config.downloader.radar_id = "KOHX"
    config.downloader.poll_interval_sec = 1
    config.downloader.latest_files = 5
    config.downloader.latest_minutes = 60
    config.downloader.start_time = None
    config.downloader.end_time = None
    config.downloader.min_file_size = 1024
    
    fake_conn = MagicMock()
    fake_conn.get_avail_radars.return_value = ["KDFW", "KLBB"]  # KOHX not in list
    
    downloader = AwsNexradDownloader(
        config=config,
        output_dir=temp_dir,
        conn=fake_conn,
    )
    
    start = datetime(2025, 12, 18, tzinfo=timezone.utc)
    end = datetime(2025, 12, 18, tzinfo=timezone.utc)
    
    with caplog.at_level(logging.WARNING):
        downloader._check_radar_available(start, end)
    
    # Should warn because check succeeded but radar not found
    assert any("Radar KOHX not found in AWS" in record.message for record in caplog.records)


def test_availability_check_does_not_warn_when_check_fails(caplog, temp_dir):
    """Do NOT warn when availability check fails (exception or all failures)."""
    config = MagicMock()
    config.downloader.radar_id = "KOHX"
    config.downloader.poll_interval_sec = 1
    config.downloader.latest_files = 5
    config.downloader.latest_minutes = 60
    config.downloader.start_time = None
    config.downloader.end_time = None
    config.downloader.min_file_size = 1024
    
    fake_conn = MagicMock()
    fake_conn.get_avail_radars.side_effect = Exception("AWS unavailable")
    
    downloader = AwsNexradDownloader(
        config=config,
        output_dir=temp_dir,
        conn=fake_conn,
    )
    
    start = datetime(2025, 12, 18, tzinfo=timezone.utc)
    end = datetime(2025, 12, 18, tzinfo=timezone.utc)
    
    with caplog.at_level(logging.WARNING):
        downloader._check_radar_available(start, end)
    
    # Should NOT warn because check failed (exception)
    assert not any("Radar KOHX not found in AWS" in record.message for record in caplog.records)


def test_availability_check_does_not_warn_when_radar_found(caplog, temp_dir):
    """Do NOT warn when radar is found in availability check."""
    config = MagicMock()
    config.downloader.radar_id = "KOHX"
    config.downloader.poll_interval_sec = 1
    config.downloader.latest_files = 5
    config.downloader.latest_minutes = 60
    config.downloader.start_time = None
    config.downloader.end_time = None
    config.downloader.min_file_size = 1024
    
    fake_conn = MagicMock()
    fake_conn.get_avail_radars.return_value = ["KDFW", "KOHX", "KLBB"]  # KOHX IS in list
    
    downloader = AwsNexradDownloader(
        config=config,
        output_dir=temp_dir,
        conn=fake_conn,
    )
    
    start = datetime(2025, 12, 18, tzinfo=timezone.utc)
    end = datetime(2025, 12, 18, tzinfo=timezone.utc)
    
    with caplog.at_level(logging.WARNING):
        downloader._check_radar_available(start, end)
    
    # Should NOT warn because radar was found
    assert not any("Radar KOHX not found in AWS" in record.message for record in caplog.records)
