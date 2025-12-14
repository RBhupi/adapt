from adapt.pipeline.processor import RadarProcessor
from pathlib import Path
import queue
import pytest

def test_process_missing_file(basic_config):
    q = queue.Queue()
    proc = RadarProcessor(q, basic_config)

    ok = proc.process_file("/does/not/exist")
    assert ok is False


def test_loader_returns_none(monkeypatch, processor_queues, basic_config):
    in_q, out_q = processor_queues
    proc = RadarProcessor(in_q, basic_config, out_q)

    monkeypatch.setattr(
        proc.loader,
        "load_and_regrid",
        lambda *a, **k: None
    )

    ok = proc.process_file("/fake/path/file")
    assert ok is False
    assert out_q.empty()

def test_processor_handles_loader_exception(monkeypatch, processor_queues, basic_config):
    in_q, out_q = processor_queues
    proc = RadarProcessor(in_q, basic_config, out_q)

    def boom(*a, **k):
        raise IOError("disk failure")

    monkeypatch.setattr(proc.loader, "load_and_regrid", boom)

    ok = proc.process_file("/fake/path/file")

    assert ok is False
    assert out_q.empty()

def test_process_missing_file(basic_config):
    proc = RadarProcessor(queue.Queue(), basic_config, queue.Queue())
    ok = proc.process_file("/does/not/exist")
    assert ok is False

import xarray as xr
import numpy as np


def test_processor_enqueues_when_load_succeeds(monkeypatch, processor_queues, basic_config):
    in_q, out_q = processor_queues
    proc = RadarProcessor(in_q, basic_config, out_q)

    fake_ds_2d = xr.Dataset(
        {"reflectivity": (("y", "x"), np.ones((4, 4)))},
        attrs={"z_level_m": 2000}
    )

    monkeypatch.setattr(
        proc,
        "_load_and_regrid",
        lambda filepath: (None, fake_ds_2d, "/fake/file.nc", None)
    )

    # bypass file existence
    monkeypatch.setattr(
        "adapt.pipeline.processor.Path.exists",
        lambda self: True
    )

    ok = proc.process_file("/fake/file")

    assert ok is True
    assert out_q.qsize() == 1
