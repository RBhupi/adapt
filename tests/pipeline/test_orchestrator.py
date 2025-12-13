from adapt.pipeline.orchestrator import PipelineOrchestrator


def test_orchestrator_initialization(basic_config):
    orch = PipelineOrchestrator(basic_config)
    assert orch.downloader_queue is not None
    assert orch.plotter_queue is not None


def test_orchestrator_logging_and_tracker(basic_config):
    orch = PipelineOrchestrator(basic_config)
    orch._setup_logging()

    assert orch.tracker is not None
    assert basic_config["file_tracker"] is orch.tracker
