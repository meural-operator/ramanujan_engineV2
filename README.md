# Ramanujan Engine: Universal Distributed Scientific Computing Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA_Ready-EE4C2C.svg)](https://pytorch.org/)
[![Distributed Compute](https://img.shields.io/badge/Computing-Distributed_Edge-yellow.svg)](https://firebase.google.com/)

A globally distributed, GPU-accelerated computing framework that orchestrates **Deep Reinforcement Learning** with **PyTorch Tensor exhaustion** to solve arbitrary scientific problems at scale. Continued Fractions discovery is just one of many pluggable modules — the framework generalizes to any problem domain.

---

## 🌟 Key Modifications

| Feature | Description |
|---|---|
| **Universal Pipeline Router** | A 4-stage abstract execution engine (`core/pipeline.py`) that decouples problem definition, search strategy, compute engine, and network coordination into fully interchangeable plugins. |
| **Deep RL Bounds Pruning** | AlphaTensor MCTS heuristic (`modules/continued_fractions/math_ai/`) intelligently slices coordinate spaces before GPU exhaustion. |
| **Modular Problem System** | Scientific problems are self-contained modules under `modules/`. Adding a new problem domain requires zero modifications to the core framework. |
| **Zero-Loss Edge Caching** | Hardened `sqlite3` local cache guarantees verified discoveries survive power/network failures. |
| **1-Click Deployment** | Windows volunteers join the cluster by double-clicking `run_node.bat` — handles Python isolation, dependencies, and credential generation automatically. |
| **Research RL Training Suite** | Dedicated Curriculum Learning PPO pipeline with TensorBoard MLOps at `research_training/`. |

---

## 🏗️ Architecture & Hierarchy

The framework is built on a strict separation between **core infrastructure** and **scientific modules**. The `UniversalPipelineRouter` orchestrates any combination of plugins without knowing their internals.

```mermaid
graph TD
    classDef core fill:#2d3436,stroke:#74b9ff,stroke-width:2px,color:#fff;
    classDef iface fill:#0984e3,stroke:#fff,stroke-width:2px,color:#fff;
    classDef module fill:#00b894,stroke:#fff,stroke-width:2px,color:#fff;
    classDef client fill:#fdcb6e,stroke:#2d3436,stroke-width:2px,color:#2d3436;

    subgraph "Core Framework"
        PIPE[UniversalPipelineRouter] ::: core
        I1[TargetProblem] ::: iface
        I2[BoundingStrategy] ::: iface
        I3[ExecutionEngine] ::: iface
        I4[NetworkCoordinator] ::: iface
    end

    subgraph "Module: Continued Fractions"
        M1[EulerMascheroniTarget] ::: module
        M2[MCTSStrategy] ::: module
        M3[CUDAEnumerator] ::: module
        M4[FirebaseCoordinator] ::: module
    end

    subgraph "Module: Future Problem"
        F1["ProteinFoldingTarget"] ::: module
        F2["GeneticStrategy"] ::: module
        F3["TPUEngine"] ::: module
    end

    CLIENT[Edge Node Client] ::: client -->|boots| PIPE
    PIPE --> I1 --> M1
    PIPE --> I2 --> M2
    PIPE --> I3 --> M3
    PIPE --> I4 --> M4

    I1 -.-> F1
    I2 -.-> F2
    I3 -.-> F3
```

### Abstract Interfaces (`core/interfaces/`)

| Interface | File | Purpose |
|---|---|---|
| `TargetProblem` | `base_problem.py` | Defines the mathematical/scientific problem — constants, verification logic, LHS hash generation |
| `BoundingStrategy` | `base_strategy.py` | Search space optimization — AI pruning, heuristics, or brute-force passthrough |
| `ExecutionEngine` | `base_engine.py` | Hardware-accelerated compute — CUDA tensors, CPU multiprocessing, TPU, etc. |
| `NetworkCoordinator` | `base_coordinator.py` | Distributed coordination — work unit fetching, result submission, authentication |

### Adding a New Scientific Module

To add a new problem domain (e.g. `protein_folding`):

```python
# modules/protein_folding/target.py
from core.interfaces.base_problem import TargetProblem

class ProteinFoldingTarget(TargetProblem):
    @property
    def name(self): return "protein-folding"

    def verify_match(self, a_coef, b_coef):
        # Your domain-specific verification
        ...
```

No changes to `core/` are required. The pipeline automatically routes through your new plugin.

---

## 🗃️ Available Modules

| Module | Directory | Status | Description |
|---|---|---|---|
| **Continued Fractions** | `modules/continued_fractions/` | ✅ Active | GPU-accelerated discovery of novel GCF formulas for mathematical constants (Euler-Mascheroni, Zeta, Catalan, etc.) |
| **RL Training Suite** | `research_training/` | ✅ Active | Curriculum PPO training for the AlphaTensor MCTS neural bounds pruner |

*Future modules can be added by implementing the 4 core interfaces — see Architecture section above.*

---

## 🚀 Execution Guide

### 1-Click Deployment (Windows Volunteers)
```bash
git clone https://github.com/meural-operator/ramanujan_engineV2.git
cd ramanujan_engineV2/clients
# Double-click run_node.bat — or from terminal:
.\run_node.bat
```
> The script auto-installs Python 3.13 via Micromamba, bootstraps all dependencies, generates Firebase credentials, seeds the LHS math tables, and launches the GPU compute node.

### Manual Research Setup
```bash
# 1. Create conda environment
conda env create -f setup/environment.yml
conda activate curiosity

# 2. Seed mathematical verification tables (one-time, ~10s)
python scripts/seed_euler_mascheroni_db.py

# 3. Launch the edge compute node
cd clients
python edge_node.py
```

### RL Neural Network Training
```bash
cd research_training
python train.py --episodes 50000 --max-depth 200

# Monitor training metrics
tensorboard --logdir runs/
```

### Running Tests
```bash
python -m unittest discover -s tests -v
```

---

## 📁 Directory Structure
```
ramanujan_engineV2/
├── core/                              # Universal Framework Engine
│   ├── interfaces/                    #   Abstract Base Classes
│   │   ├── base_problem.py            #     TargetProblem interface
│   │   ├── base_strategy.py           #     BoundingStrategy interface
│   │   ├── base_engine.py             #     ExecutionEngine interface
│   │   └── base_coordinator.py        #     NetworkCoordinator interface
│   ├── coordinators/                  #   Network I/O Implementations
│   │   └── firebase_coordinator.py    #     Firebase REST coordinator
│   └── pipeline.py                    #   UniversalPipelineRouter
│
├── modules/                           # Scientific Problem Modules
│   └── continued_fractions/           #   GCF Discovery Module
│       ├── targets/                   #     Mathematical constants
│       │   └── euler_mascheroni.py    #       Euler-Mascheroni plugin
│       ├── engines/                   #     GPU/CPU enumerators
│       │   ├── GPUEfficientGCFEnumerator.py
│       │   ├── EfficientGCFEnumerator.py
│       │   └── cuda_gcf.py           #       V4 adapter wrapper
│       ├── domains/                   #     Polynomial search spaces
│       ├── math_ai/                   #     AlphaTensor + MCTS models
│       │   ├── models/               #       Actor-Critic networks
│       │   ├── training/             #       RL training utilities
│       │   └── strategies/           #       MCTSStrategy plugin
│       └── utils/                     #     Convergence filters, etc.
│
├── clients/                           # Distributed Compute Nodes
│   ├── edge_node.py                   #   Universal client entrypoint
│   ├── run_node.bat                   #   1-click Windows deployer
│   ├── checkpoints/                   #   Compiled RL weights (.pt)
│   └── setup/                         #   Auto-installer scripts
│
├── research_training/                 # Dedicated RL Training Pipeline
│   ├── train.py                       #   PPO curriculum trainer
│   ├── config.yaml                    #   Hyperparameter config
│   └── eval_mcts.py                   #   MCTS visualizer
│
├── scripts/                           # Utility & Seeder Scripts
├── tests/                             # Unit & Integration Tests
└── README.md
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-new-module`
3. Implement your scientific module under `modules/your_module/`
4. Add tests under `tests/`
5. Submit a Pull Request

For new problem domains, implement the 4 interfaces in `core/interfaces/` and register your module. The framework handles everything else.

---

## 📄 License

This project is licensed under the MIT License.
