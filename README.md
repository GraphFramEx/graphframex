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

2. Visual Genome (optional). [Google Drive Download](https://drive.google.com/file/d/132ziPf2PKqjGoZkqh9194rT17qr3ywN8/view?usp=sharing).
   This is used for preprocessing the VG-5 dataset and visualizing the generated explanations.
   Manually download it to the same directory as `data`. (This package can be accessed by API, but we found it slow to use.) You can still run the other datasets without downloading it.

3. Other packages

```
pip install tqdm matplotlib argparse json jupyterlab notebook pgmpy captum
# For visualization (optional)
pip install tensorboardx
```

## Datasets

1. The processed raw data for datasets `syn1`, `syn2`, `syn3`, `syn4`, `syn5`, `syn6` is available in the `data/` folder.
2. Dataset `MUTAG` will be automatically downloaded when training models.
3. We select and label 4443 graphs from <https://visualgenome.org/> to construct the **VG-5** dataset. The graphs are

The dir ir aranged as

```
.
├── mutag
│   ├── mutag.pt
│   └── raw_data
│       ├── MUTAG_A.txt
│       ├── MUTAG_edge_labels.txt
│       ├── MUTAG_graph_indicator.txt
│       ├── MUTAG_graph_labels.txt
│       ├── MUTAG_node_labels.txt
│       └── README.txt
├── syn1
│   └── syn1.pt
├── syn2
│   └── syn2.pt
├── syn3
│   └── syn3.pt
├── syn4
│   └── syn4.pt
├── syn5
│   └── syn5.pt
└── syn6
    └── syn6.pt
```

## Train GNNs

We provide the trained GNNs in `model/` for reproducing the results in our paper.

## Code map

```
.
├── dataset
│   ├── __init__.py
│   ├── gen_mutag.py
│   ├── gen_syn.py
│   ├── mutag_utils.py
│   └── syn_utils
│       ├── featgen.py
│       ├── gengraph.py
│       ├── gengroundtruth.py
│       └── synthetic_structsim.py
├── evaluate
│   ├── accuracy.py
│   ├── fidelity.py
│   └── mask_utils.py
├── explainer
│   ├── __init__.py
│   ├── genmask.py
│   ├── gnnexplainer.py
│   ├── method.py
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
    ├── gen_utils.py
    ├── graph_utils.py
    ├── io_utils.py
    ├── math_utils.py
    └── parser_utils.py
```

## Explaining the Predictions

### Node Classification

```bash
python3 code/main.py --dataset [dataset-name] --explain_graph False --explainer_name [explainer_name]
```

- dataset-name: syn1, syn2, syn3, syn4, syn5, syn6
- explainer_name: random, pagerank, distance, sa_node, ig_node, gnnexplainer, subgraphx, pgmexplainer

### Graph Classification

```bash
python3 code/main.py --dataset [dataset-name] --explain_graph True --explainer_name [explainer_name]
```

- dataset-name: mutag
- explainer_name: random, sa, ig, gnnexplainer (to complete)

## Mask transformation

To compare the methods, we adopt separately three strategies to cut off the masks:

1. Sparsity

2. Threshold

3. Topk

For baseline explainers, e.g.,

```python
gnn_explainer = GNNExplainer(device, gnn_path)
gnn_explainer.explain_graph(test_dataset[0],
                           epochs=100, lr=1e-2)

screener = Screener(device, gnn_path)
screener.explain_graph(test_dataset[0])
```

5. Evaluation & Visualization

Evaluation and visualization are made universal for every `explainer`. After explaining a single graph, the pair `(graph, edge_imp:np.ndarray)` is saved as `explainer.last_result` by default, which is then evaluated or visualized.

```python
ratios = [0.1 *i for i in range(1,11)]
acc_auc = refine.evaluate_acc(ratios).mean()
racall =  refine.evaluate_recall(topk=5)
refine.visualize(vis_ratio=0.3) # visualize the explanation
```

To evaluate ReFine-FT and ReFine in the testing datasets, run

```bash
python evaluate.py --dataset ba3
```

The results will be included in file `results/ba3_results.json`, where `ReFine-FT.ACC-AUC` (`ReFine-FT.Recall@5`) and `ReFine.ACC-AUC` (`ReFine.Recall@5`) are the performances of ReFine-FT and ReFine, respectively.

## Citation

Please cite our paper if you find the repository useful.

```
@inproceedings{}
```
