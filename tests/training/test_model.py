import lightning.pytorch as pl
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.utils import dense_to_sparse

from gwnet.datasets import METRLA
from gwnet.model import GraphWavenet
from gwnet.train import GWnetForecasting


class TestTrainingStep:
    def make_datapoint(self):
        dpoint = Data()
        dpoint.x = torch.randn(207, 2, 12)
        dpoint.y = torch.randn(207, 2, 12)
        index, weights = dense_to_sparse(
            torch.nn.functional.relu(torch.randn(207, 207) - 0.3)
        )
        dpoint.edge_index = index
        dpoint.edge_attr = weights
        return dpoint

    def test_training_step(self):
        args = {"lr": 0.01, "weight_decay": 0.0001}

        batch = Batch.from_data_list([self.make_datapoint(), self.make_datapoint()])
        model = GraphWavenet(adaptive_embedding_dim=64, n_nodes=207)

        lightning_module = GWnetForecasting(args, model, missing_value=0.0)
        lightning_module.training_step(batch, 0)

    def test_trainer(self):
        args = {"lr": 0.01, "weight_decay": 0.0001}

        model = GraphWavenet(adaptive_embedding_dim=64, n_nodes=207)
        plmodule = GWnetForecasting(args, model, missing_value=0.0)

        trainer = pl.Trainer(
            # accelerator='cpu',
            max_steps=100,
            gradient_clip_val=5.0,  # TODO There was something about this in the code.
            logger=False,
        )

        dataset = METRLA("./")
        loader = DataLoader(dataset, batch_size=16, num_workers=0)

        trainer.fit(model=plmodule, train_dataloaders=loader)
