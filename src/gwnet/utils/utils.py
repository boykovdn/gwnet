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
