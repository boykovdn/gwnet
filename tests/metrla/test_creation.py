from gwnet.datasets import METRLA


class TestMETRLA:
    def test_metrla_instantiate(self):
        METRLA("./data")
