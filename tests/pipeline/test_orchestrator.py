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
