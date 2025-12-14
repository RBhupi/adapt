from adapt.pipeline.orchestrator import PipelineOrchestrator


def test_orchestrator_initialization(basic_config):
    orch = PipelineOrchestrator(basic_config)
    assert orch.downloader_queue is not None
    assert orch.plotter_queue is not None


from adapt.pipeline.orchestrator import PipelineOrchestrator
from adapt.setup_directories import setup_output_directories

def test_orchestrator_logging_and_tracker(tmp_path, basic_config):
    output_dirs = setup_output_directories(tmp_path)
    basic_config["output_dirs"] = output_dirs

    orch = PipelineOrchestrator(basic_config)
    orch._setup_logging()

    assert orch.tracker is not None
    assert basic_config["file_tracker"] is orch.tracker


def test_orchestrator_queue_wiring(basic_config):
    orch = PipelineOrchestrator(basic_config)

    assert orch.downloader_queue.maxsize == 100
    assert orch.plotter_queue.maxsize == 50

def test_orchestrator_stop_is_idempotent(basic_config):
    orch = PipelineOrchestrator(basic_config)

    orch.stop()
    orch.stop()  # should not raise
