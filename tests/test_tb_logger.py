import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Learning.ctde_ram.tb_logger import TBLogger


class RecordingWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))


def test_log_step_group_shares_then_advances_step():
    logger = TBLogger.__new__(TBLogger)
    logger.enabled = True
    logger.writer = RecordingWriter()
    logger.global_step = 7

    logger.log_step_group({"loss": 1.5, "entropy": 0.25})

    assert logger.writer.scalars == [
        ("loss", 1.5, 7),
        ("entropy", 0.25, 7),
    ]
    assert logger.global_step == 8


def test_log_step_group_advances_when_tensorboard_is_disabled():
    logger = TBLogger.__new__(TBLogger)
    logger.enabled = False
    logger.writer = None
    logger.global_step = 3

    logger.log_step_group({"loss": 1.5})

    assert logger.global_step == 4
