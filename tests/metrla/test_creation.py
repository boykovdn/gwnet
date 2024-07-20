from gwnet.datasets import METRLA

import pytest

class TestMETRLA:
    def test_metrla_instantiate(self):
        dataset = METRLA("./data")
