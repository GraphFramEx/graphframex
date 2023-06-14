# GNN Explainability Framework

The goal of GraphFramEx is to systematically evaluate methods that generate explanations
for predictions of graph neural networks (GNNs).
GraphFramEx proposes a unique metric, the characterization score,
which combines the fidelity measures, and classifies explanations
based on their quality of being sufficient or necessary.
We scope ourselves to node and graph classification tasks and
compare the most representative techniques in the field of
generative and non-generative input-level explainability for GNNs.

**Graph Classification Tasks**

| Non-generative Explainer | Paper                                                                               |
| :----------------------- | :---------------------------------------------------------------------------------- |
| Occlusion                | Visualizing and understanding convolutional networks                                |
| SA                       | Explainability Techniques for Graph Convolutional Networks.                         |
| Grad-CAM                 | Explainability Methods for Graph Convolutional Neural Networks.                     |
| Integrated Gradients     | Axiomatic Attribution for Deep Networks                                             |
| GNNExplainer             | GNNExplainer: Generating Explanations for Graph Neural Networks                     |
| SubgraphX                | On Explainability of Graph Neural Networks via Subgraph Exploration                 |
| PGM-Explainer            | PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks |

| Generative Explainer | Paper                                                                             |
| :------------------- | :-------------------------------------------------------------------------------- |
| PGExplainer          | Parameterized Explainer for Graph Neural Network                                  |
| RCExplainer          | Reinforced Causal Explainer for Graph Neural Networks                             |
| GSAT                 | Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism |
| GraphCFE             | CLEAR: Generative Counterfactual Explanations on Graphs                           |
| DiffExplainer        | D4Explainer (not published)                                                       |
| GflowExplainer       | DAG Matters! GFlowNets Enhanced Explainer For Graph Neural Networks `             |

**Node Classification Tasks**

| Explainer            | Paper                                                                               |
| :------------------- | :---------------------------------------------------------------------------------- |
| Distance             | Shortest Path Distance Approximation using Deep learning Techniques                 |
| PageRank             | The PageRank Citation Ranking: Bringing Order to the Web                            |
| Occlusion            | Visualizing and understanding convolutional networks                                |
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

1. Load every additional packages:

```
pip install -r requirements.txt
```

2. Manual installation

Pytorch Geometric. [Official Download](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

```
CUDA=cu111
TORCH=1.9.1
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric==2.0.3
```

Other packages:

```
pip install tqdm matplotlib argparse json jupyterlab notebook pgmpy captum
```

## Python code map

```
.
├── __init__.py
├── baseline.py
├── dataset
│   ├── __init__.py
│   ├── graphsst2.py
│   ├── mnist.py
│   ├── mnist_utils
│   │   ├── __init__.py
│   │   └── extract_slic.py
│   ├── mol_dataset.py
│   ├── mutag_large.py
│   ├── mutag_utils
│   │   └── gengroundtruth.py
│   ├── nc_real_dataset.py
│   ├── pow_dataset.py
│   ├── powcont_dataset.py
│   ├── powcontrnd_dataset.py
│   ├── syn_dataset.py
│   └── syn_utils
│       ├── featgen.py
│       ├── gengraph.py
│       ├── gengroundtruth.py
│       └── synthetic_structsim.py
├── evaluate
│   ├── __init__.py
│   ├── accuracy.py
│   ├── fidelity.py
│   └── mask_utils.py
├── explain.py
├── explainer
│   ├── cfgnnexplainer.py
│   ├── diffexplainer.py
│   ├── explainer_utils
│   │   ├── diffexplainer
│   │   │   ├── __init__.py
│   │   │   ├── graph_utils.py
│   │   │   └── pgnn.py
│   │   ├── gflowexplainer
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── mdp.py
│   │   │   └── sampler.py
│   │   ├── gsat
│   │   │   ├── __init__.py
│   │   │   ├── get_model.py
│   │   │   └── utils.py
│   │   └── rcexplainer
│   │       ├── __init__.py
│   │       ├── rc_train.py
│   │       └── reorganizer.py
│   ├── gflowexplainer.py
│   ├── gnnexplainer.py
│   ├── gradcam.py
│   ├── graph_explainer.py
│   ├── graphcfe.py
│   ├── gsat.py
│   ├── node_explainer.py
│   ├── pgexplainer.py
│   ├── pgmexplainer.py
│   ├── rcexplainer.py
│   ├── shapley.py
│   └── subgraphx.py
├── gendata.py
├── gnn
│   ├── __init__.py
│   ├── gnn_perturb.py
│   └── model.py
├── main.py
├── plot_mutag.py
├── train_gnn.py
└── utils
    ├── __init__.py
    ├── gen_utils.py
    ├── graph_utils.py
    ├── io_utils.py
    ├── math_utils.py
    ├── parser_utils.py
    ├── path.py
    └── plot_utils.py
```

## Run code

```bash
python3 code/main.py --dataset_name [dataset-name] --model_name [gnn-model] --explainer_name [explainer-name]
```

### Graph Classification

- dataset-name:
  - synthetic: ba_2motifs, ba_multishapes
  - real-world: mutag, bbbp, mnist, graphsst2, ieee24_mc, ieee39_mc, ieee118_mc, uk_mc
- gnn-model: gcn, gat, gin, transformer
- explainer-name: random, sa, ig, gradcam, occlusion, basic_gnnexplainer, gnnexplainer, subgraphx, pgmexplainer, pgexplainer, rcexplainer, gsat, graphcfe, gflowexplainer, diffexplainer

### Node Classification

- dataset-name:
  - synthetic: ba_house, ba_grid, tree_cycle, tree_grid, ba_bottle
  - real-world: cora, pubmed, citeseer, facebook, chameleon, squirrel, texas, wisconsin, cornell, actor
- gnn-model: gcn, gat, gin, transformer
- explainer-name: random, pagerank, distance, sa, ig, gradcam, occlusion, basic_gnnexplainer, gnnexplainer, subgraphx, pgmexplainer, pgexplainer

Note that gradcam is only available for synthetic datasets and subgraphx only for GCN model.

### Mask transformation

To compare the methods, we adopt separately three strategies to cut off the masks:

1. Sparsity

2. Threshold

3. Topk

This can be changed by changing the `--mask_transformation` parameter. Choices are [`topk`, `sparsity`,`threshold`]. The default strategy is `topk`.
You adjust the level of transformation with the `--transf_params` parameter. Here, you define the list of transformation values. Default list is `"5,10"`

### Jupyter Notebook

The default visualizations are provided in `notebook/GNN-Explainer-Viz.ipynb`.

> Note: For an interactive version, you must enable ipywidgets
>
> ```
> jupyter nbextension enable --py widgetsnbextension
> ```

Tuning the mask sparsity/threshold/top-k values.

## Dataset desciption

### Graph classification

| Dataset        |       Name       | Description                                           |
| -------------- | :--------------: | ----------------------------------------------------- |
| BA-2motifs     |   `ba_2motifs`   | Random BA graph with 2 motifs.                        |
| BA-multishapes | `ba_multishapes` | Random BA graph with multiple shapes.                 |
| MUTAG          |     `mutag`      | Mutagenecity Predicting the mutagenicity of molecules |
| BBBP           |      `bbbp`      | Blood-brain barrier penetration                       |
| MNIST          |     `mnist`      | MNIST                                                 |
| GraphSST-2     |   `graphsst2`    | GraphSST-2                                            |
| IEEE-24        |   `ieee24_mc`    | IEEE-24 (Powergrid dataset)                           |
| IEEE-39        |   `ieee39_mc`    | IEEE-39 (Powergrid dataset)                           |
| IEEE-118       |   `ieee118_mc`   | IEEE-118 (Powergrid dataset)                          |
| UK             |     `uk_mc`      | UK (Powergrid dataset)                                |

### Node classification

| Dataset          |     Name     | Description                                                                                       |
| ---------------- | :----------: | ------------------------------------------------------------------------------------------------- |
| Barabasi-House   |  `ba_house`  | Random BA graph with House attachments.                                                           |
| Barabasi-Grid    |  `ba_grid`   | Random BA graph with grid attachments.                                                            |
| Tree-Cycle       | `tree_cycle` | Random Tree with cycle attachments.                                                               |
| Tree-Grid        | `tree_grid`  | Random Tree with grid attachments.                                                                |
| Barabasi-Bottle  | `ba_bottle`  | Random BA graph with bottle attachments.                                                          |
| Cora             |    `cora`    | Citation network ([source](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)). |
| Pubmed           |   `pubmed`   | PubMed network ([source](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)).   |
| Citeseer         |  `citeseer`  | Citeseer network ([source](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)). |
| FacebookPagePage |  `facebook`  | Facebook Page-Page network dataset                                                                |
| Chameleon        | `chameleon`  | Wikipedia dataset                                                                                 |
| Squirrel         |  `squirrel`  | Wikipedia dataset                                                                                 |
| Texas            |   `texas`    | WebKB dataset                                                                                     |
| Wisconsin        | `wisconsin`  | WebKB dataset                                                                                     |
| Cornell          |  `cornell`   | WebKB dataset                                                                                     |
| Actor            |   `actor`    | Film-director-actor-writer network (Actor)                                                        |

## Citation

Please cite our [paper](https://arxiv.org/pdf/2206.09677.pdf) if you find the repository useful:

<pre><code>@article{amara2022graphframex,
  title={GraphFramEx: Towards Systematic Evaluation of Explainability Methods for Graph Neural Networks},
  author={Amara, Kenza and Ying, Rex and Zhang, Zitao and
  Han, Zhihao and Shan, Yinan and Brandes, Ulrik and Schemm, Sebastian and Zhang, Ce},
  journal={arXiv preprint arXiv:2206.09677},
  year={2022}
}</code></pre>
