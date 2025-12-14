from datetime import datetime
from pathlib import Path


def test_register_and_fetch_file(tracker):
    file_id = "KTEST_0001"
    radar_id = "KTEST"
    scan_time = datetime.utcnow()

    created = tracker.register_file(file_id, radar_id, scan_time)
    assert created is True

    status = tracker.get_file_status(file_id)
    assert status["file_id"] == file_id
    assert status["radar_id"] == radar_id
    assert status["status"] == "pending"


def test_register_duplicate_is_noop(tracker):
    file_id = "KTEST_0002"
    tracker.register_file(file_id, "KTEST", datetime.utcnow())
    created = tracker.register_file(file_id, "KTEST", datetime.utcnow())
    assert created is False


def test_stage_progression(tracker, tmp_path):
    file_id = "KTEST_0003"
    tracker.register_file(file_id, "KTEST", datetime.utcnow())

    nc = tmp_path / "grid.nc"
    tracker.mark_stage_complete(file_id, "regridded", path=nc)

    status = tracker.get_file_status(file_id)
    assert status["regridded_at"] is not None
    assert status["gridnc_path"] == str(nc)
    assert status["status"] == "processing"


def test_mark_analyzed_sets_cells(tracker):
    file_id = "KTEST_0004"
    tracker.register_file(file_id, "KTEST", datetime.utcnow())

    tracker.mark_stage_complete(file_id, "analyzed", num_cells=7)
    status = tracker.get_file_status(file_id)

    assert status["num_cells"] == 7
    assert status["status"] == "processing"


def test_mark_plotted_completes(tracker):
    file_id = "KTEST_0005"
    tracker.register_file(file_id, "KTEST", datetime.utcnow())

    tracker.mark_stage_complete(file_id, "plotted")
    status = tracker.get_file_status(file_id)

    assert status["status"] == "completed"


def test_failure_sets_failed(tracker):
    file_id = "KTEST_0006"
    tracker.register_file(file_id, "KTEST", datetime.utcnow())

    tracker.mark_stage_complete(file_id, "analyzed", error="boom")
    status = tracker.get_file_status(file_id)

    assert status["status"] == "failed"
    assert "boom" in status["error_message"]


def test_should_process_logic(tracker):
    file_id = "KTEST_0007"
    tracker.register_file(file_id, "KTEST", datetime.utcnow())

    assert tracker.should_process(file_id, "analyzed") is True

    tracker.mark_stage_complete(file_id, "analyzed")
    assert tracker.should_process(file_id, "analyzed") is False
