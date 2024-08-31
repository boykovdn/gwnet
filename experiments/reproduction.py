import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from torch_geometric.loader.dataloader import DataLoader

from gwnet.datasets import METRLA
from gwnet.model import GraphWavenet
from gwnet.train import GWnetForecasting

# from gwnet.utils import StandardScaler


def train():
    args = {"lr": 0.001, "weight_decay": 0.0001}

    model = GraphWavenet(adaptive_embedding_dim=64, n_nodes=207, k_diffusion_hops=1)
    plmodule = GWnetForecasting(args, model, missing_value=0.0)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=100,
        gradient_clip_val=5.0,  # TODO There was something about this in the code.
        logger=tb_logger,
    )

    dataset = METRLA("./")
    # import pdb; pdb.set_trace()
    # scaler = StandardScaler(dataset.z_norm_mean, dataset.z_norm_std)
    ## TODO Parametrise.
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    trainer.fit(model=plmodule, train_dataloaders=loader)


if __name__ == "__main__":
    train()
