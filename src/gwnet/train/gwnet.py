from collections.abc import Callable
from typing import Any

import lightning.pytorch as pl
import torch
from torch_geometric.data import Data

from ..utils import create_mask


class GWnetForecasting(pl.LightningModule):
    def __init__(
        self,
        args: dict[str, Any],
        model: torch.nn.Module,
        missing_value: float = 0.0,
        scaler: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        r"""
        Trains Graph wavenet for the traffic forecasting task.

        Args:
            args (dict): training args.

            model (Module): The Graph Wavenet model.

            missing_value (float): Defaults to 0.0, this is the value which
                denotes a missing value in the dataset (relevant to METR-LA),
                and no gradient is calculated to predict these values.

            scaler: A function that inverts the normalisation of the data,
                used to produce MAE losses comparable to literature.
        """
        super().__init__()

        self.args = args
        self.model = model
        self.scaler = scaler
        self.missing_value = missing_value

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.args["lr"],
            weight_decay=self.args["weight_decay"],
        )

    def masked_mae_loss(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Calculates the masked MAE loss.

        The values which match the class missing_value do not take part in
        the calculation, and thus do not propagate learning signal. This is
        done because METR-LA denotes missing values using 0.0, which is
        part of the data domain.
        """
        # create_mask handles missing_value set to None.
        mask = create_mask(targets, self.missing_value)

        num_terms = torch.sum(mask)

        loss = torch.abs(preds - targets)
        return torch.sum(loss[mask]) / num_terms

    def training_step(self, input_batch: Data, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        targets = input_batch.y
        out = self.model(input_batch)

        if self.scaler is not None:
            raise NotImplementedError()
            # out = self.scaler.inverse_transform(out)

        loss = self.masked_mae_loss(out, targets)
        self.log("train_loss", loss)

        return loss