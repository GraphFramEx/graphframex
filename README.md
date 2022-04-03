# GNN Explainability Framework

**Node Classification Tasks**

| Explainer            | Paper                                                                               |
| :------------------- | :---------------------------------------------------------------------------------- |
| PageRank             | The PageRank Citation Ranking: Bringing Order to the Web                            |
| Distance             | Shortest Path Distance Approximation using Deep learning Techniques                 |
| SA                   | Explainability Techniques for Graph Convolutional Networks.                         |
| Grad-CAM             | Explainability Methods for Graph Convolutional Neural Networks.                     |
| DeepLIFT             | Learning Important Features Through Propagating Activation Differences              |
| Integrated Gradients | Axiomatic Attribution for Deep Networks                                             |
| GNNExplainer         | GNNExplainer: Generating Explanations for Graph Neural Networks                     |
| SubgraphX            | On Explainability of Graph Neural Networks via Subgraph Exploration                 |
| PGM-Explainer        | PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks |

**Graph Classification Tasks**

| Explainer            | Paper                                                                               |
| :------------------- | :---------------------------------------------------------------------------------- |
| ReFine               | Towards Multi-Grained Explainability for Graph Neural Networks                      |
| SA                   | Explainability Techniques for Graph Convolutional Networks.                         |
| Grad-CAM             | Explainability Methods for Graph Convolutional Neural Networks.                     |
| DeepLIFT             | Learning Important Features Through Propagating Activation Differences              |
| Integrated Gradients | Axiomatic Attribution for Deep Networks                                             |
| GNNExplainer         | GNNExplainer: Generating Explanations for Graph Neural Networks                     |
| PGExplainer          | Parameterized Explainer for Graph Neural Network                                    |
| PGM-Explainer        | PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks |
| Screener             | Causal Screening to Interpret Graph Neural Networks                                 |
| CXPlain              | Cxplain: Causal Explanations for Model Interpretation under Uncertainty             |

## Installation

**Requirements**

- CPU or NVIDIA GPU, Linux, Python 3.7
- PyTorch >= 1.5.0, other packages

1. Pytorch Geometric. [Official Download](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

```
# We use TORCH version 1.6.0
CUDA=cu111
TORCH=1.9.1
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric==2.0.3
```

2. Other packages

```
pip install tqdm matplotlib argparse json jupyterlab notebook pgmpy captum
# For visualization (optional)
pip install tensorboardx
```

## Datasets

1. The processed raw data for datasets `syn1`, `syn2`, `syn3`, `syn4`, `syn5`, `syn6` is available in the `data/syn` folder.
2. The processed raw data for datasets `cora`, `citeseer`, `pubmed`, `cornell`, `texas`, `wisconsin`, `chameleon`, `squirrel`, `actor` will be automatically downloaded when training models.

## Trained GNN models

We provide the trained GNNs in `model/` for reproducing the results in our paper.

## Python code map

```
.
├── dataset
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── data_utils.cpython-38.pyc
│   │   ├── gen_planetoids.cpython-38.pyc
│   │   └── gen_syn.cpython-38.pyc
│   ├── data_utils.py
│   ├── gen_mutag.py
│   ├── gen_real.py
│   ├── gen_syn.py
│   ├── mutag_utils.py
│   └── syn_utils
│       ├── featgen.py
│       ├── gengraph.py
│       ├── gengroundtruth.py
│       └── synthetic_structsim.py
├── evaluate
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── accuracy.cpython-38.pyc
│   │   └── fidelity.cpython-38.pyc
│   ├── accuracy.py
│   ├── fidelity.py
│   └── mask_utils.py
├── explainer
│   ├── __init__.py
│   ├── genmask.py
│   ├── gnnexplainer.py
│   ├── graph_explainer.py
│   ├── node_explainer.py
│   ├── pgmexplainer.py
│   ├── shapley.py
│   └── subgraphx.py
├── gnn
│   ├── __init__.py
│   ├── eval.py
│   ├── model.py
│   └── train.py
├── main.py
└── utils
    ├── __init__.py
    ├── gen_utils.py
    ├── graph_utils.py
    ├── io_utils.py
    ├── math_utils.py
    ├── parser_utils.py
    └── plot_utils.py

```

## Demo

### Node Classification

```bash
python3 code/main.py --dataset [dataset-name] --explain_graph False --explainer_name [explainer_name]
```

- dataset-name:
  - synthetic: syn1, syn2, syn3, syn4, syn5, syn6
  - real-world: cora, pubmed, citeseer, facebook, chameleon, squirrel, texas, wisconsin, cornell, actor
- explainer_name: random, pagerank, distance, sa, ig, gradcam, occlusion, basic_gnnexplainer, gnnexplainer, subgraphx, pgmexplainer

### Graph Classification

```bash
python3 code/main.py --dataset [dataset-name] --explain_graph True --explainer_name [explainer_name]
```

- dataset-name: mutag
- explainer_name: random, sa, ig, gnnexplainer (to complete)

### Mask transformation

To compare the methods, we adopt separately three strategies to cut off the masks:

1. Sparsity

2. Threshold

3. Topk

### Baseline explainers

```python
gnn_explainer = GNNExplainer(device, gnn_path)
gnn_explainer.explain_graph(test_dataset[0],
                           epochs=100, lr=1e-2)

screener = Screener(device, gnn_path)
screener.explain_graph(test_dataset[0])
```

## Visualization

Visualisation of the GNN model training
Visualisation of the explanations

### Jupyter Notebook

The default visualizations are provided in `notebook/GNN-Explainer-Viz.ipynb`.

> Note: For an interactive version, you must enable ipywidgets
>
> ```
> jupyter nbextension enable --py widgetsnbextension
> ```

Tuningthe mask sparsity/threshold/top-k values.
You can now play around with the mask threshold in the `GNN-Explainer-Viz-interactive.ipynb`.

> TODO: Explain outputs + visualizations + baselines

#### Included experiments

| Name             | `EXPERIMENT_NAME` | Description                                                                                                                            |
| ---------------- | :---------------: | -------------------------------------------------------------------------------------------------------------------------------------- |
| Synthetic #1     |      `syn1`       | Random BA graph with House attachments.                                                                                                |
| Synthetic #2     |      `syn2`       | Random BA graph with community features.                                                                                               |
| Synthetic #3     |      `syn3`       | Random BA graph with grid attachments.                                                                                                 |
| Synthetic #4     |      `syn4`       | Random Tree with cycle attachments.                                                                                                    |
| Synthetic #5     |      `syn5`       | Random Tree with grid attachments.                                                                                                     |
| Synthetic #6     |      `syn6`       | Random BA graph with bottle attachments.                                                                                               |
| MUTAG            |      `mutag`      | Mutagenecity Predicting the mutagenicity of molecules ([source](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)). |
| Cora             |      `cora`       | Citation network ([source](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)).                                      |
| Pubmed           |     `pubmed`      | PubMed network ([source](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)).                                        |
| Citeseer         |    `citeseer`     | Citeseer network ([source](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)).                                      |
| FacebookPagePage |    `facebook`     |                                                                                                                                        |
| Chameleon        |    `chameleon`    |                                                                                                                                        |
| Squirrel         |    `squirrel`     |                                                                                                                                        |
| Texas            |      `texas`      |                                                                                                                                        |
| Wisconsin        |    `wisconsin`    |                                                                                                                                        |
| Cornell          |     `cornell`     |                                                                                                                                        |
| Actor            |      `actor`      |                                                                                                                                        |

### Using the explainer on other models

A graph convolutional model is provided. This repo is still being actively developed to support other
GNN models in the future.

## Changelog

See [CHANGELOG.md](#)

## Citation

Please cite our paper if you find the repository useful.

```
@inproceedings{}
```
