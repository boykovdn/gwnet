import torch
import logging
from torch_geometric.data import InMemoryDataset, download_url, Data
from typing import Union, Optional, Callable
import numpy as np

import pandas as pd
import os

class METRLA(InMemoryDataset):

    def __init__(self, 
                 root: Union[str, os.PathLike], 
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 ):


        self.node_filename = "METR-LA.csv"
        self.url_node = "https://zenodo.org/record/5724362/files/METR-LA.csv?download=1"
        self.adj_filename = "adj_mx_METR-LA.pkl"
        self.url_adj = (
            "https://zenodo.org/record/5724362/files/adj_mx_METR-LA.pkl?download=1"
        )
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # TODO Pickle! Eww!
        return ['METR-LA.csv', 'adj_mx_METR-LA.pkl']

    @property
    def processed_file_names(self):
        return ['metrla.pt']

    def download(self):
        logging.info("Downloading {}".format(self.url_node))
        download_url(self.url_node, self.raw_dir)
        logging.info("Downloading {}".format(self.url_adj))
        download_url(self.url_adj, self.raw_dir)
        logging.info("Downloading finished.")

    def _get_edges(self, adj: np.ndarray) -> np.ndarray:
        edge_indices = np.asarray(adj != 0).nonzero()
        return np.array([edge_indices[0], edge_indices[1]])
    
    def _get_edge_weights(self, adj):
        return adj[adj != 0]

    def _z_normalization(self, features: np.ndarray, eps: float = 1e-19) -> np.ndarray:
        mean = np.mean(features)
        std = np.std(features)

        # Save the normalization factors.
        self.z_norm_mean = mean
        self.z_norm_std = std

        return (features - mean) / (std + eps)

    def _get_targets_and_features(
            self, node_series: pd.DataFrame, num_timesteps_in: int, num_timesteps_out: int, interpolate: bool = False, normalize: bool = True
    ):
        r"""
        Build the input and output features.

        Args:
            node_series (pd.DataFrame): The input timeseries associated
                with each node.
            num_timesteps_in (int): The number of timesteps associated to
                a node and used as input features.
            num_timesteps_out (int): The number of future timesteps which
                are considered as the target prediction.
            interpolate (bool): If true, will replace the 0 values with
                an interpolation.
            normalize (bool): If true, will perform a normalization of
                the dataset to 0 mean, std 1. These values are kept in
                order to undo the normalisation if needed.
        """
        #stacked_target = self._dataset["features"].values
        stacked_target = node_series.values
        if interpolate:
            raise NotImplementedError()
            #stacked_target = self._interpolate(stacked_target)
        if normalize:
            stacked_target = self._z_normalization(stacked_target)

        # Numbers below taken from PEMS-BAY saw-tooth feature
        saw_tooth = np.linspace(-1.74648, 1.7450126, 288)
        saw_tooth = np.repeat(
            saw_tooth[:, None], np.ceil(stacked_target.shape[0] / 288), axis=1
        ).T.flatten()[: stacked_target.shape[0]]
        saw_tooth = np.repeat(saw_tooth[:, None], stacked_target.shape[1], 1)

        stacked_target = np.concatenate(
            [stacked_target[:, None, :], saw_tooth[:, None, :]], 1
        )
        features = [
            torch.from_numpy(stacked_target[i : i + num_timesteps_in, :, :].T)
            for i in range(
                stacked_target.shape[0] - num_timesteps_in - num_timesteps_out
            )
        ]
        targets = [
            torch.from_numpy(stacked_target[
                i + num_timesteps_in : i + num_timesteps_in + num_timesteps_out, 0, :
            ].T)
            for i in range(
                stacked_target.shape[0] - num_timesteps_in - num_timesteps_out
            )
        ]

        return features, targets
        
    def process(self):
        # Read data into huge `Data` list.
        # TODO
        node_ids, id_to_idx_map, adj = pd.read_pickle(self.raw_paths[1])
        edges = torch.from_numpy(self._get_edges(adj))
        edge_weights = torch.from_numpy(self._get_edge_weights(adj))

        node_series = pd.read_csv(self.raw_paths[0], index_col=0)
        features, targets = self._get_targets_and_features(node_series, 12, 12) # TODO Parametrise in/out n steps.
        data_list = [
                Data(x=features[i], 
                    edge_index=edges, 
                    edge_attr=edge_weights, 
                    y=targets[i]) for i in range(len(features))
                ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
