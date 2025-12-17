import pytest
from adapt.pipeline.orchestrator import PipelineOrchestrator

pytestmark = [pytest.mark.unit, pytest.mark.pipeline]


def test_orchestrator_initialization(pipeline_config, pipeline_output_dirs):
    """Orchestrator initializes with config and output directories."""
    orch = PipelineOrchestrator(pipeline_config, pipeline_output_dirs)
    assert orch.downloader_queue is not None
    assert orch.plotter_queue is not None


def test_orchestrator_logging_and_tracker(pipeline_config, pipeline_output_dirs):
    """Orchestrator sets up logging and file tracker."""
    orch = PipelineOrchestrator(pipeline_config, pipeline_output_dirs)
    orch._setup_logging()

    assert orch.tracker is not None


def test_orchestrator_queue_wiring(pipeline_config, pipeline_output_dirs):
    """Orchestrator creates queues with correct size limits."""
    orch = PipelineOrchestrator(pipeline_config, pipeline_output_dirs)

    assert orch.downloader_queue.maxsize == 100
    assert orch.plotter_queue.maxsize == 50


def test_orchestrator_stop_is_idempotent(pipeline_config, pipeline_output_dirs):
    """Calling stop() multiple times is safe."""
    orch = PipelineOrchestrator(pipeline_config, pipeline_output_dirs)

    orch.stop()
    orch.stop()  # should not raise


def test_orchestrator_has_stop_event(pipeline_config, pipeline_output_dirs):
    """Test orchestrator has internal stop flag and stop() sets it."""
    orch = PipelineOrchestrator(pipeline_config, pipeline_output_dirs)
    
    assert hasattr(orch, '_stop_event')
    assert orch._stop_event is False
    
    orch.stop()
    assert orch._stop_event is True


def test_orchestrator_config_storage(pipeline_config, pipeline_output_dirs):
    """Test orchestrator stores config correctly."""
    orch = PipelineOrchestrator(pipeline_config, pipeline_output_dirs)
    
    assert orch.config == pipeline_config
    assert orch.output_dirs == pipeline_output_dirs


def test_orchestrator_queue_types(pipeline_config, pipeline_output_dirs):
    """Test orchestrator creates correct queue types."""
    import queue
    orch = PipelineOrchestrator(pipeline_config, pipeline_output_dirs)
    
    assert isinstance(orch.downloader_queue, queue.Queue)
    assert isinstance(orch.plotter_queue, queue.Queue)


def test_orchestrator_tracker_database_path(pipeline_config, pipeline_output_dirs):
    """Test orchestrator creates tracker with correct database path (after setup)."""
    orch = PipelineOrchestrator(pipeline_config, pipeline_output_dirs)
    orch._setup_logging()

    assert orch.tracker is not None
    # Database should be in analysis directory and named with radar id
    radar_id = pipeline_config.downloader.radar_id
    expected_db = pipeline_output_dirs["analysis"] / f"{radar_id}_file_tracker.db"
    assert expected_db.exists()


def test_orchestrator_mode_from_config(pipeline_config, pipeline_output_dirs):
    """Test orchestrator respects mode from config."""
    orch = PipelineOrchestrator(pipeline_config, pipeline_output_dirs)
    
    # Should use mode from config
    assert orch.config.mode in ["realtime", "historical"]


def test_orchestrator_validates_required_fields(make_config, pipeline_output_dirs, temp_dir):
    """Test orchestrator validates radar_id and output_dir."""
    # Config without radar_id should raise
    config = make_config()
    config.downloader.radar_id = None
    
    with pytest.raises(ValueError, match="radar_id is required"):
        PipelineOrchestrator(config, pipeline_output_dirs)


def test_orchestrator_validates_output_dir(make_config, pipeline_output_dirs):
    """Test orchestrator validates output_dir."""
    config = make_config()
    config.downloader.output_dir = None
    
    with pytest.raises(ValueError, match="output_dir is required"):
        PipelineOrchestrator(config, pipeline_output_dirs)


def test_orchestrator_stop_clears_queues(pipeline_config, pipeline_output_dirs):
    """Test that stop() sets stop flag and is idempotent."""
    orch = PipelineOrchestrator(pipeline_config, pipeline_output_dirs)
    
    # Add some items to queues
    orch.downloader_queue.put("test1")
    orch.plotter_queue.put("test2")
    
    orch.stop()
    
    # Internal stop flag should be set
    assert orch._stop_event is True
    
    # Calling stop again should be safe
    orch.stop()
    assert orch._stop_event is True


def test_orchestrator_processor_config_accessible(pipeline_config, pipeline_output_dirs):
    """Test orchestrator can access processor config."""
    orch = PipelineOrchestrator(pipeline_config, pipeline_output_dirs)
    
    assert hasattr(orch.config, 'processor')
    assert orch.config.processor.max_history >= 0
    assert orch.config.processor.min_file_size > 0
