# GNN Explainability Framework

**Node Classification Tasks**

| Explainer            | Paper                                                                               |
| :------------------- | :---------------------------------------------------------------------------------- |
| Distance             | Shortest Path Distance Approximation using Deep learning Techniques                 |
| PageRank             | The PageRank Citation Ranking: Bringing Order to the Web                            |
| SA                   | Explainability Techniques for Graph Convolutional Networks.                         |
| Grad-CAM             | Explainability Methods for Graph Convolutional Neural Networks.                     |
| Integrated Gradients | Axiomatic Attribution for Deep Networks                                             |
| GNNExplainer         | GNNExplainer: Generating Explanations for Graph Neural Networks                     |
| PGExplainer          | Parameterized Explainer for Graph Neural Network                                    |
| SubgraphX            | On Explainability of Graph Neural Networks via Subgraph Exploration                 |
| PGM-Explainer        | PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks |

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
```

## Datasets

1. The processed raw data for datasets `ba_house`, `ba_community`, `ba_grid`, `tree_cycle`, `tree_grid`, `ba_bottle` is available in the `data/syn` folder.
2. The processed raw data for datasets `cora`, `citeseer`, `pubmed`, `cornell`, `texas`, `wisconsin`, `chameleon`, `squirrel`, `actor` will be automatically downloaded when training models.

## Python code map

```
.
├── dataset
│   ├── __init__.py
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
│   ├── pgexplainer.py
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

## Node Classification

```bash
python3 code/main.py --dataset [dataset-name] --explain_graph False --explainer_name [explainer_name]
```

- dataset-name:
  - synthetic: ba_house, ba_grid, tree_cycle, tree_grid, ba_bottle
  - real-world: cora, pubmed, citeseer, facebook, chameleon, squirrel, texas, wisconsin, cornell, actor
- explainer_name: random, pagerank, distance, sa, ig, gradcam, occlusion, basic_gnnexplainer, gnnexplainer, subgraphx, pgmexplainer, pgexplainer

Note that gradcam is only available for synthetic datasets.

### Mask transformation

To compare the methods, we adopt separately three strategies to cut off the masks:

1. Sparsity

2. Threshold

3. Topk

This can be changed by changing the `--strategy` parameter. Choices are [`topk`, `sparsity`,`threshold`]. The default strategy is `topk`.
You adjust the level of transformation with the `--params_list` parameter. Here, you define the list of transformation values. Default list is `"5,10"`

### Jupyter Notebook

The default visualizations are provided in `notebook/GNN-Explainer-Viz.ipynb`.

> Note: For an interactive version, you must enable ipywidgets
>
> ```
> jupyter nbextension enable --py widgetsnbextension
> ```

Tuning the mask sparsity/threshold/top-k values.

#### Included experiments

| Name             | `EXPERIMENT_NAME` | Description                                                                                                                            |
| ---------------- | :---------------: | -------------------------------------------------------------------------------------------------------------------------------------- |
| Barabasi-House   |    `ba_house`     | Random BA graph with House attachments.                                                                                                |
| Barabasi-Grid    |     `ba_grid`     | Random BA graph with grid attachments.                                                                                                 |
| Tree-Cycle       |   `tree_cycle`    | Random Tree with cycle attachments.                                                                                                    |
| Tree-Grid        |    `tree_grid`    | Random Tree with grid attachments.                                                                                                     |
| Barabasi-Bottle  |    `ba_bottle`    | Random BA graph with bottle attachments.                                                                                               |
| MUTAG            |      `mutag`      | Mutagenecity Predicting the mutagenicity of molecules ([source](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)). |
| Cora             |      `cora`       | Citation network ([source](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)).                                      |
| Pubmed           |     `pubmed`      | PubMed network ([source](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)).                                        |
| Citeseer         |    `citeseer`     | Citeseer network ([source](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)).                                      |
| FacebookPagePage |    `facebook`     | Facebook Page-Page network dataset                                                                                                     |
| Chameleon        |    `chameleon`    | Wikipedia dataset                                                                                                                      |
| Squirrel         |    `squirrel`     | Wikipedia dataset                                                                                                                      |
| Texas            |      `texas`      | WebKB dataset                                                                                                                          |
| Wisconsin        |    `wisconsin`    | WebKB dataset                                                                                                                          |
| Cornell          |     `cornell`     | WebKB dataset                                                                                                                          |
| Actor            |      `actor`      | Film-director-actor-writer network (Actor)                                                                                             |

### Using the explainer on other models

A graph convolutional model is provided. This repo is still being actively developed to support other
GNN models in the future.

## Citation

Please cite our paper if you find the repository useful.
