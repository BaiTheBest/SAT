from typing import Optional

import torch
from torch import Tensor


class History_Series(torch.nn.Module):
    r"""A historical embedding storage module."""
    def __init__(self, graph_size: int, emd_dim: int):
        super().__init__()

        self.history_series = []
        
        self.reset_parameters()

    def reset_parameters(self):
        self.history_series = []

    def push(self, new_embeddings):
        self.history_series.append(new_embeddings)

    def pop(self, index=0):
        del self.history_series[index]

    def pull(self, idx=None):
        out = self.history_series
        if idx != None:
            out = self.history_series[idx]
        return out

    def push_embeddings(self, node_id, emb, idx=-1):
        self.history_series[idx][node_id] = emb

    @property
    def length(self):
        return len(self.history_series)




    