<h1 align="center">SAT: Staleness-Alleviated Training Framework</h1>

This repo is the practical realization of our *<ins>S</ins>taleness-<ins>A</ins>lleviated <ins>T</ins>raining* (*SAT*) framework, which reduces the embedding staleness adaptively, as described in our paper:

**Staleness-Alleviated Distributed Graph Neural Network Training via Online
Dynamic-Embedding Prediction** 



*SAT* is implemented in [PyTorch](https://pytorch.org/) and utilizes the [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) (PyG) library.

A detailed description of *SAT* can be found in its implementation.

## Requirements

* Install [**PyTorch >= 1.7.0**](https://pytorch.org/get-started/locally/)
* Install [**PyTorch Geometric >= 1.7.0**](https://github.com/rusty1s/pytorch_geometric#installation):

```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

where `${TORCH}` should be replaced by either `1.7.0` or `1.8.0`, and `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101`, `cu102`, `cu110` or `cu111`, depending on your PyTorch installation.

## Installation

```
python setup.py install
```

## Project Structure

* **`torch_geometric_autoscale/`** contains the source code of *SAT*
* **`small_benchmark/`** includes experiments to evaluate *SAT* performance on *small-scale* graphs

We use [**Hydra**](https://hydra.cc/) to manage hyperparameter configurations.
