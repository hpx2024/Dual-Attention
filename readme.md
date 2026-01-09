# A General Neural Backbone for Mixed-Integer Linear Optimization via Dual Attention

This repository provides the official implementation of our paper: ["A General Neural Backbone for Mixed-Integer Linear Optimization via Dual Attention"](https://arxiv.org/abs/2601.04509)

![Model Overview](img/overview.png)

## Overview

Mixed-Integer Linear Programming (MILP) is a fundamental approach for combinatorial optimization with applications across science and engineering. However, solving large-scale MILP instances remains computationally challenging.

Recent advances in deep learning have addressed this challenge by representing MILP instances as bipartite graphs and using Graph Neural Networks (GNNs) to extract structural patterns. While promising, GNN-based approaches are inherently limited by their local message-passing mechanisms, restricting their representation power for complex MILP problems.

This repository introduces an attention-driven neural model that learns richer representations beyond conventional graph-based methods. Our key contributions include:

- Dual Attention Mechanism: Our model employs parallel self-attention and cross-attention modules for variables and constraints, enabling global information exchange and deeper representation learning.

- Versatile Architecture: The proposed model serves as a general-purpose backbone applicable to multiple downstream tasks:
    - Instance-level prediction
    - Element-level prediction
    - Solution state prediction

- State-of-the-Art Performance: Extensive experiments on widely-used benchmarks demonstrate consistent improvements over existing baselines, establishing our attention-based model as a powerful foundation for learning-enhanced MILP optimization.

## Usage

This repository is organized into three main tasks, corresponding to the three levels of MILP solving:

### Repository Structure
```
Dual-Attention-Backbone/
├── 01.Instance-Level/
├── 02.Element-Level/
├── 03.Solving-State-Level/
└── img/
```

### Quick Start

Each task directory contains:
- Data generation scripts
- Training scripts
- Evaluation scripts
- Task-specific readme with detailed instructions

To reproduce results for any task:

1. Navigate to the task directory:
```bash
   cd 01.Instance-Level  # or 02.Element-Level, 03.Solving-State-Level
```

2. Follow the task-specific readme for:
   - Environment setup
   - Dataset generation
   - Model training
   - Evaluation

For detailed instructions, please refer to the readme in each task directory.


## Citation

If you find our work helpful, please cite:
```bibtex
@article{huang2026general,
  title={A General Neural Backbone for Mixed-Integer Linear Optimization via Dual Attention},
  author={Huang, Peixin and Wu, Yaoxin and Ma, Yining and Wu, Cathy and Song, Wen and Zhang, Wei},
  journal={arXiv preprint arXiv:2601.04509},
  year={2026}
}
```