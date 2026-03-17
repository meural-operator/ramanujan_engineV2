# Ramanujan Machine Engine V2

The Ramanujan Machine is an algorithmic approach to discover new mathematical conjectures. This project focuses on number theory, specifically finding formulas relating fundamental constants like pi, e, the Riemann zeta function values, and the Euler-Mascheroni constant to various continued fractions.

For background information, please visit [RamanujanMachine.com](https://www.RamanujanMachine.com).

## 🚀 Engine V2: AI & GPU Architecture

This iteration of the Ramanujan Machine Engine introduces a major architectural overhaul, transforming the search engine into a high-performance, AI-guided discovery platform capable of evaluating billions of combinations seamlessly.

### Key Cutting-Edge Features

1. **Modular Deep RL Framework (`ramanujan/math_ai/`)**:
   - **Gym-Native Environments**: Abstracted Gym-like RL environments (`AbstractRLEnvironment`, `GCFRewardEnvironment`) where the "reward" is the matched digits of numerical convergence.
   - **Actor-Critic Neural Models**: PyTorch-based neural networks that evaluate deep mathematical sequence trajectories to predict optimal Upper Confidence Bounds (UCB) for search domains.

2. **Neural-Guided MCTS Search (`NeuralMCTSPolyDomain`)**:
   - Instead of relying on brute-force combinatorics or heavy-handed gradient descent, the Engine now deploys AlphaTensor-style Deep Reinforcement Learning.
   - The AI simulates random mathematical traversals, scoring them with the Actor-Critic network, and returning a narrowly optimized, mathematically dense search space for the GPU.

3. **High-Performance GPU Exhaustion**:
   - **`GPUEfficientGCFEnumerator`**: Takes the bound predictions from the AI and generates massive mathematical tensor arrays on standard NVIDIA CUDA hardware. 
   - **Zero-Latency Hash Matching**: Replaced massive standard Python iteration loops with heavily vectorized `torch.isin` intersections directly on the CUDA cores.

4. **Asynchronous CPU Verification (Dual Threading)**:
   - The engine operates on a Producer-Consumer threading model. The GPU (Producer) blasts through millions of combinations a second.
   - When a preliminary hit is found, it is instantly dropped into a thread-safe Queue. 
   - A concurrent background CPU thread (Consumer) runs high-precision `mpmath` validations continuously. The GPU never blocks or waits for the CPU.

## Installation

Clone the repository and install the dependencies. The Engine requires PyTorch and basic CUDA support to fully execute the V2 pipeline.

```bash
conda create -n curiosity python=3.13
conda activate curiosity
pip install -e .
```

## Running the Code

To start a new execution, you must define the LHS (Left Hand Side) constant, the search Domain, and the Enumerator.

### Example: Deep RL Euler-Mascheroni Search

Here is how you initialize the AI-driven search pipeline (as seen in `scripts/euler_mascheroni_research_grade.py`):

#### 1. Generate LHS Hash Table
Create the hash boundary matrix for the target mathematical constant.
```python
from ramanujan.LHSHashTable import LHSHashTable
from ramanujan.constants import g_const_dict

const_val = g_const_dict['euler-mascheroni']
lhs = LHSHashTable('euler_mascheroni.db', 30, [const_val])
```

#### 2. Initialize the Neural MCTS Domain
Use the AI to discover the optimal iteration boundaries.
```python
from ramanujan.poly_domains.NeuralMCTSPolyDomain import NeuralMCTSPolyDomain

poly_search_domain = NeuralMCTSPolyDomain(
    a_deg=3, a_coef_range=[-16, 16],
    b_deg=3, b_coef_range=[-16, 16],
    target_val=const_val,
    mcts_simulations=1000 # Let the Actor-Critic network run 1k sequence rollouts
)
```

#### 3. Start the Asynchronous GPU/CPU Enumerator
Execute the highly-parallelized enumeration.
```python
from ramanujan.enumerators.GPUEfficientGCFEnumerator import GPUEfficientGCFEnumerator

enumerator = GPUEfficientGCFEnumerator(lhs, poly_search_domain, [const_val])

# Output will display a Dual-TQDM layout.
# 1. GPU iteration bounds (processing ~20M/s)
# 2. CPU verification queue (processing arbitrary precision maths)
results = enumerator.full_execution()

enumerator.print_results(results)
```

## Legacy Support

While `Engine V2` brings Deep RL, the architecture is entirely backwards-compatible and Plug-and-Play. You can still use simple Cartesian bounds or basic Monte Carlo samplings while taking full advantage of the Zero-Latency GPU backend.

```python
# Just swap the AI Domain out for a simple baseline domain:
from ramanujan.poly_domains.CartesianProductPolyDomain import CartesianProductPolyDomain

poly_search_domain = CartesianProductPolyDomain(
    a_deg=2, a_coef_range=[-5, 5],
    b_deg=2, b_coef_range=[-5, 5]
)
# The GPUEfficientGCFEnumerator will still execute it asynchronously.
```
