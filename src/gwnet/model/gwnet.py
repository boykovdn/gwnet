from collections import OrderedDict
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

# from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils.sparse import dense_to_sparse

from .layer import TempGCN


class GatedTCN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        dilation: int = 1,
    ) -> None:
        r"""
        The GatedTCN module from Graph Wavenet.
        """
        super().__init__()
        self.filter_conv = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.gate_conv = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, batch: Data) -> Data:
        r"""
        This module uses Conv1d, so the sequence dimension is expected (*, C, L),
        for channels C, sequence length L.
        """
        assert len(batch.x.shape) == 3, f"Expected (N,C,L), got {batch.shape}"
        # C == in_channels
        N, C, L = batch.x.shape

        # The Conv can be done concurrently across each node with dimensionality C,
        # series len L.
        out = self.tanh(self.filter_conv(batch.x)) * self.sigmoid(
            self.gate_conv(batch.x)
        )
        # Last dimension is inferred, it will be the sequence length reduced.
        batch.x = out
        return batch


class DiffusionConv(torch.nn.Module):
    def __init__(
        self,
        args: dict[str, Any],
        in_channels: int,
        out_channels: int,
        # k_hops: int = 3,
        bias: bool = True,
    ) -> None:
        r"""
        Args:
            args (dict): Parameters from the parent modules.

            in_channels (int): Input dimensionality

            out_channels (int): Output dimensionality

            k_hops (int): The highest power of the transition matrix. This is
                equivalent to the number of hops away signal is propagated from.
        """
        super().__init__()

        # self.id_linear = torch.nn.Linear(in_channels, out_channels, bias=bias)

        # Create one GCN per hop and diffusion direction.
        gcn_dict = {}
        if args["fwd"]:
            for k in range(1, args["k_hops"] + 1):
                gcn_dict[f"fwd_{k}"] = TempGCN(
                    in_channels, out_channels, bias=False, node_dim=0
                )

        if args["bck"]:
            for k in range(1, args["k_hops"] + 1):
                gcn_dict[f"bck_{k}"] = TempGCN(
                    in_channels, out_channels, bias=False, node_dim=0
                )

        if args["adp"]:
            for k in range(1, args["k_hops"] + 1):
                gcn_dict[f"adp_{k}"] = TempGCN(
                    in_channels, out_channels, bias=False, node_dim=0
                )

        if bias:
            self.bias = Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias", None)

        self.gcns = torch.nn.ModuleDict(gcn_dict)

    def forward(self, x: Data, cached_params: dict[str, torch.Tensor]) -> Data:
        r"""
        Args:
            x (Data): Input graph.
        """
        out_sum = 0
        for adj_name, adj_index in cached_params["adj_indices"].items():
            gcn_ = self.gcns[adj_name]
            # NOTE Some trickery here. In TempGCN the dense layer works on the
            # last dimension. The layer is also configured to aggregate along
            # the 0th dim rather than the default -2. x.x is (N, C, L), so we
            # transpose C_in, L then transpose back to C_out, L.
            out_sum += gcn_(
                x.x.transpose(1, 2), adj_index, cached_params["adj_weights"][adj_name]
            ).transpose(1, 2)

        x.x = out_sum

        if self.bias is not None:
            x.x += self.bias.view(1, -1, 1)

        return x


class STResidualModule(torch.nn.Module):
    def __init__(
        self,
        args: dict[str, Any],
        in_channels: int,
        interm_channels: int,
        out_channels: int,
        dilation: int = 1,
        kernel_size: int = 2,
        disable_gcn: bool = False,
    ):
        r"""
        Wraps the TCN and GCN modules.

        Args:
            args (dict): Contains parameters passed from the parent module, namely
                flags showing which adjacency matrices to expect and initialise
                parameters for.

            disable_gcn (bool): If True, the GCN aggregation will not be computed,
                so effectively the model will have no graph component.
        """
        super().__init__()

        self._disable_gcn = disable_gcn

        self.tcn = GatedTCN(
            in_channels, interm_channels, kernel_size=kernel_size, dilation=dilation
        )
        self.gcn = DiffusionConv(
            args, interm_channels, out_channels
        )  # TODO # interm -> out channels, diffusion_hops

    def forward(self, x: Data, cached_adj: dict[str, torch.Tensor]) -> Data:
        r"""
        Apply the gated TCN followed by GCN.

        Note that TCN will act on each node across the batches independently,
        so we can reshape to merge the nodes into the batch dimension. When
        applying GCN, we will need to keep the batch and node dimension
        separate, because the adjacency matrix will define which nodes talk
        to each other, and the node index matters.

        Args:
            x (Data): The batched Data object.

            cached_adj (dict): Adjacency matrices used in the GCN calculation.
        """
        # TCN works on the features alone, and handles the (N,C,L) shape
        # internally.
        tcn_out = self.tcn(x)

        if self._disable_gcn:
            return tcn_out

        return self.gcn(tcn_out, cached_adj)


class GraphWavenet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 12,
        dilation_channels: int = 32,
        residual_channels: int = 32,
        skip_channels: int = 256,
        end_channels: int = 512,
        n_layers: int = 8,
        dilations_override: None | list[int] = None,
        k_diffusion_hops: int = 3,
        adaptive_embedding_dim: int | None = None,
        n_nodes: int | None = None,
        forward_diffusion: bool = True,
        backward_diffusion: bool = True,
        disable_gcn: bool = False,
    ):
        r"""
        Initialise the GWnet model.

        It requires the road network adjacency matrix, because it runs
        on the entire road network for every prediction. It always
        predicts for all nodes at once. The powers of the adjacency
        used in the diffusion layer can be pre-computed, hence the adj
        is required at instantiation.
        """
        super().__init__()

        if adaptive_embedding_dim is not None:
            if n_nodes is None:
                adp_err_msg = "If adaptive_embedding_dim is passed, \
                    n_nodes should also be passed."
                raise Exception(adp_err_msg)

            self.node_embeddings = torch.nn.Parameter(
                torch.rand(n_nodes, adaptive_embedding_dim)
            )
            adp = True

        else:
            self.register_parameter("node_embeddings", None)
            adp = False

        # This model accepts the entire road network, hence can cache
        # the diffusion adjacency matrices, doesn't have to take their
        # powers on every forward pass.

        # Some of these parameters
        self.global_elements: dict[str, Any] = {
            "fwd": forward_diffusion,
            "bck": backward_diffusion,
            "adp": adp,
            "adj_indices": None,
            "adj_weights": None,
            "k_hops": k_diffusion_hops,
        }

        # Set the dilations to the default in the paper, unless different
        # ones are given.
        dilations = [1, 2, 1, 2, 1, 2, 1, 2]
        if dilations_override:
            assert len(dilations_override) == n_layers
            dilations = dilations_override

        # Initialize the linear modules.
        # NOTE For Linear, the channels should come last.
        self.init_linear = torch.nn.Linear(in_channels, residual_channels)

        # Initialize the residual layers.
        res_layers = []
        skip_convs = []
        for idx_layer in range(n_layers):
            dilation = dilations[idx_layer]
            res_layers.append(
                (
                    f"residual_layer_{idx_layer}",
                    STResidualModule(
                        self.global_elements,
                        residual_channels,
                        dilation_channels,
                        residual_channels,
                        dilation=dilation,
                        disable_gcn=disable_gcn,
                    ),
                )
            )

            skip_convs.append(
                (
                    f"skip_conv_{idx_layer}",
                    torch.nn.Conv1d(residual_channels, skip_channels, kernel_size=1),
                )
            )

        self.residual_net = torch.nn.Sequential(OrderedDict(res_layers))
        self.skip_convs = torch.nn.Sequential(OrderedDict(skip_convs))

        self.relu = torch.nn.ReLU()
        self.out_linear_0 = torch.nn.Linear(skip_channels, end_channels)
        self.out_linear_1 = torch.nn.Linear(end_channels, out_channels)

    def _update_adp_adj(self, batch_size: int, k_hops: int) -> None:
        r"""
        Update the adaptive adjacency matrix.

        Uses the node embeddings to compute an adjacency matrix using inner
        products.

        Args:
            batch_size (int): The input batch size. This is needed to
                expand the adjacency as a block diagonal matrix across
                the batch.
        """

        if (
            self.global_elements["adj_indices"] is None
            or self.global_elements["adj_weights"] is None
        ):
            self.global_elements["adj_indices"] = {}
            self.global_elements["adj_weights"] = {}

        # (N, C) @ (C, N) -> (N, N)
        adp_adj = F.softmax(
            F.relu(self.node_embeddings @ self.node_embeddings.T), dim=1
        )
        adp_adj_dense_batch = torch.block_diag(*[adp_adj] * batch_size)

        for k in range(1, k_hops + 1):
            adp_dense_power = adp_adj_dense_batch**k
            adp_indices, adp_weights = dense_to_sparse(adp_dense_power)
            self.global_elements["adj_indices"][f"adp_{k}"] = adp_indices
            self.global_elements["adj_weights"][f"adp_{k}"] = adp_weights

    def _precompute_gcn_adjs(
        self,
        adj: torch.Tensor,
        weights: torch.Tensor,
        k_hops: int,
        fwd: bool = True,
        bck: bool = True,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        r"""
        Return the adj matrix along with powers of it.

        Args:
            adj (torch.Tensor): The edge index of the dataset

            weights (torch.Tensor): The edge attrs (weights) of the graph.

            k_hops (int): If k_hops is 1, then the one-hop neighbourhood is included,
                along with the identity. If k_hops is 0, then the adjacencies are an
                empty list.

            fwd (bool): Whether to compute the powers of the forward transition matrix.

            bck (bool): Whether to compute the powers of the backward transition matrix.

        Returns:
            dict, dict
        """
        adj_dense = to_dense_adj(adj, edge_attr=weights)[0]

        # Compute the transition matrices from the adjacency.
        transition_dense = adj_dense / adj_dense.sum(1).repeat((len(adj_dense), 1)).T

        index_dict = {}
        weight_dict = {}
        for k in range(1, k_hops + 1):
            if fwd:
                adj_dense_power = transition_dense**k
                adj_sparse_indices, adj_sparse_weights = dense_to_sparse(
                    adj_dense_power
                )
                index_dict[f"fwd_{k}"] = adj_sparse_indices
                weight_dict[f"fwd_{k}"] = adj_sparse_weights

            if bck:
                adj_T_dense_power = transition_dense.T**k
                adj_T_sparse_indices, adj_T_sparse_weights = dense_to_sparse(
                    adj_T_dense_power
                )
                index_dict[f"bck_{k}"] = adj_T_sparse_indices
                weight_dict[f"bck_{k}"] = adj_T_sparse_weights

        return index_dict, weight_dict

    def forward(self, batch: Data) -> torch.Tensor:
        r"""
        Run the forward pass of the GraphWavenet model.

        The input data comes as a batch of Data objects, but throughout the
        execution of the model, the features might be reshaped and operated
        on by torch Modules. The GCN module requires the adjacency matrix
        and works on the Data batch itself.

        Args:
            batch: Input batch
        """
        batched_index, batched_weights = batch.edge_index, batch.edge_attr

        # The powers of the graph transition matrix are computed once and
        # stored globally.
        if (
            self.global_elements["adj_indices"] is None
            and self.global_elements["adj_weights"] is None
        ):
            self.global_elements["adj_indices"], self.global_elements["adj_weights"] = (
                self._precompute_gcn_adjs(
                    batched_index,
                    batched_weights,
                    self.global_elements["k_hops"],
                    fwd=self.global_elements["fwd"],
                    bck=self.global_elements["bck"],
                )
            )

        if self.node_embeddings is not None:
            self._update_adp_adj(batch.batch.max() + 1, self.global_elements["k_hops"])

        # x_dict = batch.x_dict
        # edge_index_dict = batch.edge_index_dict
        # edge_attr_dict = (
        #    batch.edge_attr_dict
        # )  # TODO Should I check if this is present first?

        # Here transpose (N, C, L) -> (N, L, C) then back.
        # Linear expects channels at the end.
        batch.x = self.init_linear(batch.x.transpose(1, 2)).transpose(1, 2)

        x_skipped: None | torch.Tensor = None
        x_intermetiate: None | torch.Tensor = None
        for layer_idx in range(len(self.residual_net)):
            residual_layer = self.residual_net[layer_idx]
            skip_conv = self.skip_convs[layer_idx]

            if x_intermetiate is None:
                # keep interlayer result.
                x_intermediate = residual_layer(batch, self.global_elements)
            else:
                # Update the dict to record the latest layer's output.
                x_intermediate = residual_layer(x_intermediate, self.global_elements)

            if x_skipped is None:
                # Create the output tensor that will store the sum of the skip
                # connections.
                # TODO Make sure I am using Conv1d correctly, and that L has to be
                # the final dimension.
                # import pdb; pdb.set_trace()
                # TODO Off the top of my head, you take the final sequence element
                # and add to the skip vector.
                # TODO Should check this.
                # NOTE Selecting final element, then expanding the axis back out.
                # TODO Can probably replace this Conv1D with Linear, should be more
                # more readable.
                x_skipped = skip_conv(x_intermediate.x[..., -1][..., None])[..., 0]
            else:
                x_skipped = (
                    x_skipped + skip_conv(x_intermediate.x[..., -1][..., None])[..., 0]
                )

        x_out = self.relu(x_skipped)
        x_out = self.out_linear_0(x_out)
        x_out = self.relu(x_out)
        return self.out_linear_1(x_out)
