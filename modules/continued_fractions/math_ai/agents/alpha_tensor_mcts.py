"""
Research-Grade Neural MCTS (Monte Carlo Tree Search) Agent.

Implements a proper MCTS tree with:
  - MCTSNode: explicit tree node structure tracking N, W, Q, P, children
  - UCB-1 selection with PUCT formula (AlphaZero style)
  - Dirichlet noise injection at root for exploration diversity
  - Global min-max Q normalization for numerically stable UCB
  - O(1) state restoration via environment snapshots (no path replay)
  - Policy improvement: action selected proportionally to visit count distribution
  - Gaussian log-probability priors for principled UCB exploration weighting
"""
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from modules.continued_fractions.math_ai.models.actor_critic import ActorCriticGCFNetwork
from modules.continued_fractions.math_ai.environments.AbstractRLEnvironment import AbstractRLEnvironment


class MCTSNode:
    """
    A single node in the MCTS tree.
    Stores:
      - state: the environment observation at this node
      - env_state: serialized environment snapshot for O(1) restoration
      - parent / action_from_parent: tree structure
      - N: visit count
      - W: cumulative value (sum of backed-up values)
      - Q: mean value W/N
      - P: prior probability from the neural network (for UCB)
      - children: dict mapping action_key → MCTSNode
    """
    __slots__ = ('state', 'env_state', 'parent', 'action_from_parent',
                 'N', 'W', 'Q', 'P', 'children', 'is_terminal')

    def __init__(self, state: np.ndarray, parent: Optional['MCTSNode'] = None,
                 action_from_parent: Optional[np.ndarray] = None, prior: float = 1.0,
                 env_state: Optional[dict] = None):
        self.state = state
        self.env_state = env_state          # Snapshot for O(1) state restoration
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.N: int = 0
        self.W: float = 0.0
        self.Q: float = 0.0
        self.P: float = prior               # prior probability from policy network
        self.children: Dict[int, 'MCTSNode'] = {}
        self.is_terminal: bool = False

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def update(self, value: float):
        """Backpropagate a value estimate up from a leaf."""
        self.N += 1
        self.W += value
        self.Q = self.W / self.N


class AlphaTensorMCTS:
    """
    Neural-Guided MCTS for GCF Discovery.
    
    The search loop (standard AlphaZero):
      1. SELECT: traverse tree using UCB-PUCT until a leaf node is reached
      2. EXPAND: generate n_actions candidate children from the neural policy prior
      3. EVALUATE: use critic V(leaf) as the value estimate
      4. BACKUP: propagate V from LEAF upward through the full ancestor path to root
    
    State Management:
      Each MCTSNode stores an env_state snapshot captured at expansion time.
      This eliminates the O(depth × N_simulations) path replay overhead — 
      restoring a node's state is a single O(1) set_state() call instead of
      replaying every ancestor action from root.
    
    Prior Computation:
      Priors are computed as the Gaussian log-probability of each sampled action
      under the policy distribution N(mean, std). This properly differentiates
      actions close to the policy mean (high prior → less exploration needed)
      from outlier actions (low prior → explored only when exploitation is saturated).
    
    After `num_simulations` iterations, the visit count distribution at the root
    defines the improved policy π̂(a|s_root) used for PPO training targets.
    """

    def __init__(self, env: AbstractRLEnvironment, network: ActorCriticGCFNetwork,
                 num_simulations: int = 200, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25,
                 n_actions: int = 8):
        self.env = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = network.to(self.device)
        self.network.eval()

        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.n_actions = n_actions

        # Global min/max for Q normalization (updated during tree search)
        self._q_min = float('inf')
        self._q_max = float('-inf')
        
        # Cache the policy std for search radius computation (Issue #7)
        self._last_policy_std = None

    def _normalize_q(self, q: float) -> float:
        """Min-max normalize Q to [0,1] for numerically stable UCB computation."""
        if self._q_max > self._q_min:
            return (q - self._q_min) / (self._q_max - self._q_min + 1e-8)
        return q

    def _ucb_score(self, node: MCTSNode, child: MCTSNode) -> float:
        """
        UCB-PUCT score (AlphaZero formulation):
          UCB = Q_norm + c_puct * P * sqrt(N_parent) / (1 + N_child)
        """
        q_norm = self._normalize_q(child.Q)
        exploration = self.c_puct * child.P * (np.sqrt(node.N) / (1 + child.N))
        return q_norm + exploration

    def _select(self, node: MCTSNode) -> Tuple[MCTSNode, List[MCTSNode]]:
        """
        Traverse tree selecting max UCB child until a leaf is found.
        Returns (leaf_node, full_path_from_root_to_leaf).
        """
        path = [node]
        while not node.is_leaf() and not node.is_terminal:
            best_score = -float('inf')
            best_child = None
            for child in node.children.values():
                score = self._ucb_score(node, child)
                if score > best_score:
                    best_score = score
                    best_child = child
            node = best_child
            path.append(node)
        return node, path

    @torch.no_grad()
    def _get_policy_value(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Query neural network for policy parameters and value estimate.
        Returns (mean, std, sampled_actions_with_log_priors, value).
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean, std, value = self.network(state_t)
        mean = mean.cpu().numpy().squeeze(0)
        std = std.cpu().numpy().squeeze(0)
        value = value.cpu().item()
        
        # Cache std for get_action_for_bounds radius computation
        self._last_policy_std = std.copy()

        # Sample n_actions candidate actions from the policy distribution
        actions = np.random.normal(mean, std, size=(self.n_actions, len(mean)))
        
        # Compute priors as Gaussian log-probability under the policy distribution.
        # P(a | μ, σ) ∝ exp(-0.5 * Σ((a_i - μ_i) / σ_i)²)
        # This properly weights actions near the policy mode higher than outliers,
        # giving UCB-PUCT a meaningful exploration-exploitation tradeoff.
        std_safe = np.clip(std, 1e-6, None)  # Prevent division by zero
        log_probs = -0.5 * np.sum(((actions - mean) / std_safe) ** 2, axis=1)
        # Softmax normalization to get valid probability distribution
        log_probs -= log_probs.max()  # numerical stability
        priors = np.exp(log_probs)
        priors /= priors.sum() + 1e-8

        return actions, priors, std, value

    def _expand(self, node: MCTSNode) -> float:
        """
        Expand a leaf node: generate n_actions children using policy network.
        Each child stores the env state snapshot for O(1) restoration.
        
        Returns:
            The critic's value estimate V(leaf) for backup.
        """
        actions, priors, std, value = self._get_policy_value(node.state)

        # Inject Dirichlet noise at root for exploration
        if node.parent is None:
            noise = np.random.dirichlet([self.dirichlet_alpha] * self.n_actions)
            priors = (1 - self.dirichlet_epsilon) * priors + self.dirichlet_epsilon * noise

        for i, (action, prior) in enumerate(zip(actions, priors)):
            # Restore to parent state before each child step
            if node.env_state is not None:
                self.env.set_state(node.env_state)
            
            obs, reward, done, _ = self.env.step(action)
            child_env_state = self.env.get_state()
            
            child = MCTSNode(
                state=np.array(obs, dtype=np.float32),
                parent=node,
                action_from_parent=action,
                prior=float(prior),
                env_state=child_env_state,
            )
            child.is_terminal = done
            node.children[i] = child

        return value

    def _backup(self, path: List[MCTSNode], value: float):
        """
        Backpropagate the value estimate through the FULL path from leaf to root.
        
        Standard AlphaZero backup: every node in the selection path gets updated,
        not just the expanded node and its immediate children.
        """
        for node in reversed(path):
            node.update(value)
            # Update global min/max Q for normalization
            if node.Q < self._q_min:
                self._q_min = node.Q
            if node.Q > self._q_max:
                self._q_max = node.Q

    def search(self, initial_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute num_simulations MCTS simulations from the given initial state.
        
        Returns:
            best_action: The single highest visit-count action (for inference)
            visit_probs: Normalized visit count distribution over root's children
                         (useful as training target for the policy network)
        """
        # Reset global Q bounds for this search
        self._q_min = float('inf')
        self._q_max = float('-inf')

        # Reset environment and capture root state snapshot
        self.env.reset()
        root_env_state = self.env.get_state()

        # Build root node with env state snapshot
        root = MCTSNode(
            state=np.array(initial_state, dtype=np.float32),
            env_state=root_env_state,
        )

        for _ in range(self.num_simulations):
            # 1. SELECT — traverse tree to a leaf, tracking full path
            leaf, path = self._select(root)

            if leaf.is_terminal:
                # Terminal node: backup 0 through the path
                self._backup(path, 0.0)
                continue

            # 2. EXPAND — create children of leaf
            # 3. EVALUATE — V(leaf) is returned by _expand
            value = self._expand(leaf)

            # 4. BACKUP — propagate V(leaf) through the FULL selection path
            self._backup(path, value)

        # Extract visit count distribution over root children
        if not root.children:
            default_action = np.zeros(2, dtype=np.float32)
            return default_action, np.array([1.0])

        child_visits = np.array([child.N for child in root.children.values()], dtype=np.float32)
        child_actions = [child.action_from_parent for child in root.children.values()]

        visit_probs = child_visits / (child_visits.sum() + 1e-8)

        # Best action = highest visit count (most explored → most confident)
        best_idx = int(np.argmax(child_visits))
        best_action = child_actions[best_idx]

        return best_action, visit_probs

    def get_action_for_bounds(self, initial_state: np.ndarray,
                               original_a_range: List, original_b_range: List,
                               n_sigma: float = 2.0) -> Tuple[List, List]:
        """
        High-level API for NeuralMCTSPolyDomain integration.
        Runs MCTS and converts the best continuous action into integer GCF bounds.
        
        The search radius is derived from the policy network's predicted standard
        deviation (σ), not an arbitrary multiplier. The bounds cover ±n_sigma
        standard deviations around the best action's implied center, ensuring
        the search region has a principled relationship to the network's uncertainty.
        
        Args:
            initial_state: Starting GCF trajectory observation
            original_a_range: Current a_n per-coefficient bounds (list of [min, max])
            original_b_range: Current b_n per-coefficient bounds (list of [min, max])
            n_sigma: Number of policy std deviations to cover (default: 2.0 = ~95% CI)
        
        Returns:
            Updated (a_coef_range, b_coef_range) with tightened bounds
        """
        best_action, _ = self.search(initial_state)
        
        # Use the policy network's own uncertainty (std) to derive search radius.
        # If the network is confident (small std), the search region is tight.
        # If uncertain (large std), we search more broadly.
        if self._last_policy_std is not None and len(self._last_policy_std) >= 2:
            a_radius = max(2, int(abs(self._last_policy_std[0]) * n_sigma))
            b_radius = max(2, int(abs(self._last_policy_std[1]) * n_sigma))
        else:
            # Fallback: use 20% of original range width
            a_widths = [hi - lo for lo, hi in original_a_range]
            b_widths = [hi - lo for lo, hi in original_b_range]
            a_radius = max(2, int(np.mean(a_widths) * 0.2))
            b_radius = max(2, int(np.mean(b_widths) * 0.2))

        # Center the search radius around the midpoint of original bounds,
        # shifted by the best action's direction signal
        a_shift = int(np.clip(best_action[0], -a_radius, a_radius))
        b_shift = int(np.clip(best_action[1], -b_radius, b_radius))

        new_a_range = []
        for lo, hi in original_a_range:
            mid = (lo + hi) // 2 + a_shift
            new_a_range.append([max(lo, mid - a_radius), min(hi, mid + a_radius)])

        new_b_range = []
        for lo, hi in original_b_range:
            mid = (lo + hi) // 2 + b_shift
            new_b_range.append([max(lo, mid - b_radius), min(hi, mid + b_radius)])

        return new_a_range, new_b_range
