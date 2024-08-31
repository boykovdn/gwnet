import torch
from torch_geometric.nn import MessagePassing


class TempGCN(MessagePassing):
    def __init__(  # type: ignore[no-untyped-def]
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        **kwargs,
    ) -> None:
        r"""
        Very cut-down version of GCNConv that works on (N, C, L) inputs.

        This class is built, because the provided modules from PyG don't
        seem to be happy working on the feature tensors associated with
        the nodes of GraphWavenet. They seem to instead have been built
        for (N, C) feature tensors, where N is the number of nodes, and
        C is the dimensionality of the feature vectors. In our case, we
        work with (N, C, L), where L is the sequence length of the agglo-
        merated time series features.

        Args:
            in_channels (int): The input channels, C_in in (N, C_in, L)

            out_channels (int): The output channels, C_out in (N, C_out, L)

            bias (bool): Whether to add a bias term to the linear trans-
                form. Default to False, because in my use case I handle
                the bias in a separate layer.

            **kwargs: Options for MessagePassing.
        """
        kwargs.setdefault("aggr", "add")

        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear = torch.nn.Linear(in_channels, out_channels, bias=bias)

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        # Message is the original input times the edge weight.
        return x_j * edge_weight.view(-1, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Run the TempGCN layer.

        Note here that x has a nonstandard dimensionality.

        Args:
            x (torch.Tensor): (N, C, L) node features.

            edge_index (torch.Tensor): (2, N) Edge index.

            edge_weight (torch.Tensor): (N, 1)
        """
        z = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return self.linear(z)
