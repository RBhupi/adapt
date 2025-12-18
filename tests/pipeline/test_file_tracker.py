import pytest
from datetime import datetime, timezone
from pathlib import Path

pytestmark = [pytest.mark.unit, pytest.mark.pipeline]


def test_register_and_fetch_file(tracker):
    file_id = "KTEST_0001"
    radar_id = "KTEST"
    scan_time = datetime.now(timezone.utc)

    created = tracker.register_file(file_id, radar_id, scan_time)
    assert created is True

    status = tracker.get_file_status(file_id)
    assert status["file_id"] == file_id
    assert status["radar_id"] == radar_id
    assert status["status"] == "pending"


def test_register_duplicate_is_noop(tracker):
    file_id = "KTEST_0002"
    tracker.register_file(file_id, "KTEST", datetime.now(timezone.utc))
    created = tracker.register_file(file_id, "KTEST", datetime.now(timezone.utc))
    assert created is False


def test_stage_progression(tracker, tmp_path):
    file_id = "KTEST_0003"
    tracker.register_file(file_id, "KTEST", datetime.now(timezone.utc))

    nc = tmp_path / "grid.nc"
    tracker.mark_stage_complete(file_id, "regridded", path=nc)

    status = tracker.get_file_status(file_id)
    assert status["regridded_at"] is not None
    assert status["gridnc_path"] == str(nc)
    assert status["status"] == "processing"


def test_mark_analyzed_sets_cells(tracker):
    file_id = "KTEST_0004"
    tracker.register_file(file_id, "KTEST", datetime.now(timezone.utc))

    tracker.mark_stage_complete(file_id, "analyzed", num_cells=7)
    status = tracker.get_file_status(file_id)

    assert status["num_cells"] == 7
    assert status["status"] == "processing"


def test_mark_plotted_completes(tracker):
    file_id = "KTEST_0005"
    tracker.register_file(file_id, "KTEST", datetime.now(timezone.utc))

    tracker.mark_stage_complete(file_id, "plotted")
    status = tracker.get_file_status(file_id)

    assert status["status"] == "completed"


def test_failure_sets_failed(tracker):
    file_id = "KTEST_0006"
    tracker.register_file(file_id, "KTEST", datetime.now(timezone.utc))

    tracker.mark_stage_complete(file_id, "analyzed", error="boom")
    status = tracker.get_file_status(file_id)

    assert status["status"] == "failed"
    assert "boom" in status["error_message"]


def test_should_process_logic(tracker):
    file_id = "KTEST_0007"
    tracker.register_file(file_id, "KTEST", datetime.now(timezone.utc))

    assert tracker.should_process(file_id, "analyzed") is True

    tracker.mark_stage_complete(file_id, "analyzed")


def test_get_pending_files(tracker):
    """Test getting pending files."""
    file_id1 = "KTEST_0008"
    file_id2 = "KTEST_0009"
    
    tracker.register_file(file_id1, "KTEST", datetime.now(timezone.utc))
    tracker.register_file(file_id2, "KTEST", datetime.now(timezone.utc))
    tracker.mark_stage_complete(file_id1, "downloaded")
    
    pending = tracker.get_pending_files()
    assert any(f["file_id"] == file_id2 for f in pending)


def test_get_statistics(tracker):
    """Test tracker statistics retrieval."""
    tracker.register_file("KTEST_0010", "KTEST", datetime.now(timezone.utc))
    tracker.register_file("KTEST_0011", "KTEST", datetime.now(timezone.utc))
    tracker.mark_stage_complete("KTEST_0010", "downloaded")
    
    stats = tracker.get_statistics()
    assert stats["total"] >= 2
    assert stats["pending"] >= 1
    assert stats["processing"] >= 1


def test_reset_failed_files(tracker):
    """Test resetting failed files."""
    file_id = "KTEST_0012"
    tracker.register_file(file_id, "KTEST", datetime.now(timezone.utc))
    tracker.mark_stage_complete(file_id, "analyzed", error="Download error")
    
    tracker.reset_failed("KTEST")
    status = tracker.get_file_status(file_id)
    assert status["status"] == "pending"


def test_mark_multiple_stages(tracker, tmp_path):
    """Test marking file through multiple stages."""
    file_id = "KTEST_0016"
    tracker.register_file(file_id, "KTEST", datetime.now(timezone.utc))
    
    tracker.mark_stage_complete(file_id, "downloaded")
    assert tracker.get_file_status(file_id)["status"] == "processing"
    
    nc = tmp_path / "grid.nc"
    tracker.mark_stage_complete(file_id, "regridded", path=nc)
    assert tracker.get_file_status(file_id)["gridnc_path"] == str(nc)
    
    tracker.mark_stage_complete(file_id, "analyzed", num_cells=5)
    assert tracker.get_file_status(file_id)["num_cells"] == 5
    
    tracker.mark_stage_complete(file_id, "plotted")
    assert tracker.get_file_status(file_id)["status"] == "completed"


def test_cleanup_deleted_files(tracker):
    """Test cleanup of deleted files."""
    file_id = "KTEST_0017"
    tracker.register_file(file_id, "KTEST", datetime.now(timezone.utc))
    
    # This method should not raise
    tracker.cleanup_deleted_files("KTEST")


def test_get_statistics_by_radar(tracker):
    """Test getting statistics for specific radar."""
    tracker.register_file("KTEST_0018", "KTEST", datetime.now(timezone.utc))
    tracker.register_file("KMOB_0001", "KMOB", datetime.now(timezone.utc))
    
    stats = tracker.get_statistics(radar_id="KTEST")
    assert stats["total"] >= 1

    pending = tracker.get_pending_files(radar_id="KTEST")
    assert any(f["file_id"] == "KTEST_0018" for f in pending)
    # Newly registered file should need processing for 'analyzed' (no timestamps set)
    assert tracker.should_process("KTEST_0018", "analyzed") is True
