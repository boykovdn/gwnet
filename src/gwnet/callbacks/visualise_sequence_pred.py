from copy import deepcopy

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class VisualiseSequencePrediction(Callback):
    def __init__(self, device: str = "cpu") -> None:
        r"""
        Args:
            device (str): Device on which execution takes place, should
                be the same as the one the model is on.
        """
        self.device = torch.device(device)
        self.true_added = False
        self.seq_counter = 0

    @torch.no_grad()  # type: ignore[misc]
    def visualise_sequence(self, trainer: Trainer, pl_module: LightningModule) -> None:
        node_idx = 0
        seq_len = 288
        offset = 300
        steps_ahead = 11

        dset = trainer.val_dataloaders.dataset
        model = pl_module.model

        for t in range(offset, offset + seq_len):
            data = dset[t]
            # Add dummy batch array of all 0s. This corresponds to all nodes
            # being part of the same batch.
            data.batch = torch.zeros(data.x.shape[0]).int()
            data.ptr = torch.zeros(1).int()
            data = data.to(self.device)

            # Two ugly conditionals make sure that the model works on
            # scaled data, and the output is scaled back into mph.
            if pl_module.scaler is not None:
                data.x[:, 0] = pl_module.scaler.transform(data.x[:, 0])

            # Deepcopy the input data, because the mode changes the x
            # tensor in-place throughout the feedforward network.
            out = model(deepcopy(data))

            if pl_module.scaler is not None:
                out = pl_module.scaler.inverse_transform(out)

            if not self.true_added:
                if pl_module.scaler is not None:
                    # Invert the data x to mph.
                    data.x[:, 0] = pl_module.scaler.inverse_transform(data.x[:, 0])

                trainer.logger.experiment.add_scalar(
                    "Sequence true", data.y[node_idx, steps_ahead], t
                )

            trainer.logger.experiment.add_scalar(
                f"Sequence_pred/{self.seq_counter:05d}", out[node_idx, steps_ahead], t
            )

        self.seq_counter += 1
        self.true_added = True

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.visualise_sequence(trainer, pl_module)
