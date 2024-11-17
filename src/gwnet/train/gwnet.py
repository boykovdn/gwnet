from typing import Any

import lightning.pytorch as pl
import torch
from torch_geometric.data import Data

from ..utils import TrafficStandardScaler, create_mask


class GWnetForecasting(pl.LightningModule):
    def __init__(
        self,
        args: dict[str, Any],
        model: torch.nn.Module,
        missing_value: float = 0.0,
        scaler: TrafficStandardScaler | None = None,
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

        if num_terms == 0:
            # Occasionally, all values are missing or 0. In this case,
            # return a loss of 0 and gradient function, which can be
            # done by selecting no values (mask all False) and summing.
            return torch.sum(loss[mask])

        return torch.sum(loss[mask]) / num_terms

    def validation_step(self, input_batch: Data, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        if self.scaler is not None:
            # NOTE Normalise only the traffic feature, hardcoded as 0.
            input_batch.x[:, 0] = self.scaler.transform(input_batch.x[:, 0])

        targets = input_batch.y
        out = self.model(input_batch)

        if self.scaler is not None:
            out = self.scaler.inverse_transform(out)

        loss = self.masked_mae_loss(out, targets)

        if loss != 0.0:
            # A loss of 0.0 means all values are missing or 0. This pollutes the log.
            self.log("val_loss", loss)

        return loss

    def training_step(self, input_batch: Data, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        if self.scaler is not None:
            # NOTE Normalise only the traffic feature, hardcoded as 0.
            input_batch.x[:, 0] = self.scaler.transform(input_batch.x[:, 0])

        targets = input_batch.y
        out = self.model(input_batch)

        if self.scaler is not None:
            out = self.scaler.inverse_transform(out)

        loss = self.masked_mae_loss(out, targets)

        if loss != 0.0:
            # A loss of 0.0 means all values are missing or 0. This pollutes the log.
            self.log("train_loss", loss)

        return loss
