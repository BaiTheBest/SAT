from typing import Tuple


import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


def get_planetoid(root: str, name: str) -> Tuple[Data, int, int]:
    transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
    dataset = Planetoid(f'{root}/Planetoid', name, transform=transform)
    return dataset[0], dataset.num_features, dataset.num_classes


def get_data(root: str, name: str) -> Tuple[Data, int, int]:
    if name.lower() in ['cora']:
        return get_planetoid(root, name)
    else:
        raise NotImplementedError
