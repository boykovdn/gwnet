import lightning.pytorch as pl
from torch_geometric.loader.dataloader import DataLoader

from gwnet.datasets import METRLA
from gwnet.model import GraphWavenet
from gwnet.train import GWnetForecasting


def train():
    args = {"lr": 0.01, "weight_decay": 0.0001}

    model = GraphWavenet(adaptive_embedding_dim=64, n_nodes=207, k_diffusion_hops=1)
    plmodule = GWnetForecasting(args, model, missing_value=0.0)

    trainer = pl.Trainer(
        accelerator="mps",
        max_steps=100,
        gradient_clip_val=5.0,  # TODO There was something about this in the code.
    )

    dataset = METRLA("./")
    ## TODO Parametrise.
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    trainer.fit(model=plmodule, train_dataloaders=loader)


if __name__ == "__main__":
    train()
