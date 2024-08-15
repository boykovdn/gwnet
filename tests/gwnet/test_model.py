import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dense_to_sparse

from gwnet.model import GraphWavenet


class TestModel:
    def make_datapoint(self):
        dpoint = Data()
        dpoint.x = torch.randn(207, 2, 12)
        index, weights = dense_to_sparse(
            torch.nn.functional.relu(torch.randn(207, 207) - 0.3)
        )
        dpoint.edge_index = index
        dpoint.edge_attr = weights
        return dpoint

    def test_creation(self):
        GraphWavenet()

    def test_forward(self):
        batch = Batch.from_data_list([self.make_datapoint(), self.make_datapoint()])

        model = GraphWavenet(adaptive_embedding_dim=64, n_nodes=207)
        model(batch)
