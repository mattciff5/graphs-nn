from typing import Any, Mapping
import torch as pt
import torch_geometric as tg

from dyn_graph_classifier import DynGraphClassifier
from dyn_graph_learner import DynGraphLearner
from ops import *
from utils import get_x_split


class DBGSLearner(pt.nn.Module):

    """
    Dynamic brain graph structure learner
    """

    def __init__(self,
                 cfg: Mapping[str, Any]) -> None:

        super().__init__()
        self.cfg = cfg
        self.dyn_graph_learner = DynGraphLearner(cfg)
        self.dyn_graph_classifier = DynGraphClassifier(cfg)
        # self.fc = pt.nn.Linear(cfg.gcn_d, cfg.n_classes, bias=True)
        self.mlp = pt.nn.Sequential(
            pt.nn.Linear(cfg.gcn_d * cfg.n_neurons, 1024),  # 1024 for example
            pt.nn.ReLU(),  
            pt.nn.Linear(1024, cfg.n_neurons * 490)
        )

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        x_split = get_x_split(self.cfg, x)
        # print('pre node_features', pt.isnan(x_split))
        node_features, sparse_adjacency, edge_index_batch, edge_attr_batch, batch = self.dyn_graph_learner(x_split)
        gru_input = pt.transpose(node_features, 1, 2).reshape(self.cfg.batch_size*self.cfg.n_neurons, self.cfg.t_repetition, self.cfg.n_neurons)
        # print('pre dyn_graph_classifier', pt.isnan(gru_input))
        out = self.dyn_graph_classifier(gru_input, edge_index_batch, edge_attr_batch, batch)
        out = out.reshape(self.cfg.batch_size, self.cfg.n_neurons, self.cfg.t_repetition, self.cfg.gcn_d)
        # out = pt.sum(out, (1, 2))
        out = pt.sum(out, (2))
        out = out.reshape(self.cfg.batch_size, -1)
        # out = self.fc(out)
        # out = pt.softmax(out, -1)
        out = self.mlp(out)
        # Reshape to original input dimensions
        out = out.view(-1, self.cfg.n_neurons, 490)
        return out