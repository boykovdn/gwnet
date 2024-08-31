from __future__ import annotations

from random import randint

import torch
from torch_geometric.data import Dataset
from tqdm import tqdm


def create_mask(
    matrix: torch.TensorType, null_value: None | float = 0.0
) -> torch.Tensor:
    if null_value is None:
        return torch.ones_like(matrix).bool()

    # NOTE Casting to float in order to handle integer values.
    return ~torch.isclose(
        matrix.float() - torch.tensor(null_value).float(), torch.tensor(0.0)
    )


class TrafficStandardScaler:
    def __init__(self, mu: float, std: float) -> None:
        self.mu = mu
        self.std = std

    @classmethod
    def from_dataset(
        cls, dataset: Dataset, n_samples: int = 100
    ) -> TrafficStandardScaler:
        traffic_vals: torch.Tensor | list[torch.Tensor] = []  # Holds tensors for stack.
        for _ in tqdm(range(n_samples), desc="Initialising scaler statistics..."):
            randidx = randint(0, len(dataset) - 1)
            # NOTE Here 0th feature is hardcoded as the traffic, unravel
            # into a sequence of values to be computed over.
            traffic_vals.append(dataset[randidx].x[:, 0, :].ravel())

        traffic_vals = torch.stack(traffic_vals)
        # NOTE The missing values are denoted as 0, don't include in mean computation.
        mu = traffic_vals[traffic_vals != 0].mean()
        std = traffic_vals[traffic_vals != 0].std()

        return cls(mu, std)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) / self.std

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mu
