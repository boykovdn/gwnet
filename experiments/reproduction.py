import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from torch_geometric.loader.dataloader import DataLoader

from gwnet.datasets import METRLA
from gwnet.model import GraphWavenet
from gwnet.train import GWnetForecasting
from gwnet.utils import TrafficStandardScaler


def train():
    args = {"lr": 0.001, "weight_decay": 0.0001}
    device = "gpu"
    num_workers = 0  # NOTE Set to 0 for single thread debugging!

    model = GraphWavenet(adaptive_embedding_dim=64, n_nodes=207, k_diffusion_hops=1)
    plmodule = GWnetForecasting(args, model, missing_value=0.0)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    trainer = pl.Trainer(
        accelerator=device,
        max_steps=10000,
        gradient_clip_val=5.0,  # TODO There was something about this in the code.
        logger=tb_logger,
    )

    dataset = METRLA("./")
    scaler = TrafficStandardScaler.from_dataset(dataset, n_samples=30000)
    plmodule.scaler = scaler
    ## TODO Parametrise.
    loader = DataLoader(dataset, batch_size=32, num_workers=num_workers)

    trainer.fit(model=plmodule, train_dataloaders=loader)


if __name__ == "__main__":
    train()
