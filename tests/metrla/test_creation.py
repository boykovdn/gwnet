from gwnet.datasets import METRLA


class TestMETRLA:
    def test_metrla_instantiate(self):
        METRLA("./data")

    def test_metrla_read(self):
        dataset = METRLA("./data")
        dataset[0]
