import queue
from pathlib import Path

from adapt.pipeline.processor import RadarProcessor
from adapt.setup_directories import setup_output_directories
from tests.utils.fake_grid import make_fake_grid_ds


def test_processor_accepts_fake_grid(tmp_path, monkeypatch, basic_config):
    output_dirs = setup_output_directories(tmp_path)
    basic_config["output_dirs"] = output_dirs

    in_q = queue.Queue()
    out_q = queue.Queue()

    proc = RadarProcessor(in_q, basic_config, out_q)

    fake_grid = make_fake_grid_ds(with_labels=True)

    # ---- Patch loader ----
    monkeypatch.setattr(
        proc.loader,
        "load_and_regrid",
        lambda *a, **k: fake_grid
    )

    # ---- Patch filesystem ----
    monkeypatch.setattr(
        "adapt.pipeline.processor.Path.exists",
        lambda self: True
    )

    ok = proc.process_file("/fake/file")

    assert ok is True

