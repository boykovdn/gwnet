import lightning.pytorch as pl
import torch
from lightning.pytorch import loggers as pl_loggers
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader.dataloader import DataLoader

from gwnet.callbacks import VisualiseSequencePrediction
from gwnet.datasets import METRLA
from gwnet.model import GraphWavenet
from gwnet.train import GWnetForecasting
from gwnet.utils import TrafficStandardScaler


def split_train_test_val(dataset: InMemoryDataset):
    r"""
    Splits the Dataset into 70:10:20 train:test:val split.

    The exact ratios are chosen to follow the literature (check Graph Wavenet paper).
    """
    len_dset = len(dataset)

    n_train = int(len_dset * 0.7)
    n_test = int(len_dset * 0.1)

    return (
        dataset[:n_train],
        dataset[n_train : (n_train + n_test)],
        dataset[(n_train + n_test) :],
    )


def train():
    args = {"lr": 0.001, "weight_decay": 0.0001}
    device = "cuda"
    num_workers = 0  # NOTE Set to 0 for single thread debugging!
    train_batch_size = 32
    val_batch_size = 16
    scaler_num_samples = 20000

    model = GraphWavenet(
        adaptive_embedding_dim=None, n_nodes=207, k_diffusion_hops=3, disable_gcn=False
    )
    plmodule = GWnetForecasting(args, model, missing_value=0.0)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    trainer = pl.Trainer(
        accelerator=device,
        max_steps=10000,  # 100000,
        limit_val_batches=1,  # TODO Debugging
        gradient_clip_val=5.0,  # TODO There was something about this in the code.
        logger=tb_logger,
        val_check_interval=500,
        callbacks=[VisualiseSequencePrediction(torch.device(device))],
    )

    dataset = METRLA("./")
    train_dataset, test_dataset, val_dataset = split_train_test_val(dataset)
    scaler = TrafficStandardScaler.from_dataset(
        train_dataset, n_samples=scaler_num_samples
    )
    plmodule.scaler = scaler

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=val_batch_size, num_workers=num_workers, shuffle=True
    )

    # This is for debugging the visualisation atm.
    trainer.fit(
        model=plmodule, train_dataloaders=train_loader, val_dataloaders=val_loader
    )


if __name__ == "__main__":
    train()
