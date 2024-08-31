import torch


def create_mask(
    matrix: torch.TensorType, null_value: None | float = 0.0
) -> torch.Tensor:
    if null_value is None:
        return torch.ones_like(matrix).bool()

    # NOTE Casting to float in order to handle integer values.
    return ~torch.isclose(
        matrix.float() - torch.tensor(null_value).float(), torch.tensor(0.0)
    )


class StandardScaler:
    def __init__(self, mu: torch.Tensor, std: torch.Tensor) -> None:
        self.mu = mu
        self.std = std

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) / self.std

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mu
