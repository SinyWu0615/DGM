from typing import DefaultDict
from collections import defaultdict
from torch.functional import Tensor
from torch_geometric.data import Data
from utils.utilities import fixed_unigram_candidate_sampler
import torch,sys
import numpy as np
import torch_geometric as tg
import scipy.sparse as sp

import config_file
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, time_steps, graphs, features, adjs):
        super(MyDataset, self).__init__()
        self.graphs = graphs
        self.features = features
        self.adjs = [self._normalize_graph_gcn(a)  for a  in adjs]
        self.time_steps = time_steps
        self.train_nodes = list(self.graphs[self.time_steps-1].nodes()) # all nodes in the graph.
        self.pyg_graphs = self._build_pyg_graphs()
        self.__createitems__()

    def _normalize_graph_gcn(self, adj):
        """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
        adj = sp.coo_matrix(adj, dtype=np.float32)
        adj_ = adj + sp.eye(adj.shape[0], dtype=np.float32)
        rowsum = np.array(adj_.sum(1), dtype=np.float32)
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten(), dtype=np.float32)
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return adj_normalized

    def _preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        features = np.array(features.todense())
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features


    def _build_pyg_graphs(self):
        pyg_graphs = []
        for feat, adj in zip(self.features, self.adjs):
            x = feat
            edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj)
            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
            pyg_graphs.append(data)
        return pyg_graphs

    def __len__(self):
        return len(self.train_nodes)

    def __getitem__(self, index):
        node = self.train_nodes[index]
        return self.data_items[node]
    
    def __createitems__(self):
        self.data_items = {}
        for node in list(self.graphs[self.time_steps-1].nodes()):
            feed_dict = {}
            feed_dict["graphs"] = self.pyg_graphs
        
            self.data_items[node] = feed_dict

    @staticmethod
    def collate_fn(samples):
        batch_dict = {}
        batch_dict["graphs"] = samples[0]["graphs"]
        return batch_dict


    
