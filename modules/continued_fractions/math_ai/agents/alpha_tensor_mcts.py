"""
Research-Grade Neural MCTS (Monte Carlo Tree Search) Agent.

Implements a proper MCTS tree with:
  - MCTSNode: explicit tree node structure tracking N, W, Q, P, children
  - UCB-1 selection with PUCT formula (AlphaZero style)
  - Dirichlet noise injection at root for exploration diversity
  - Global min-max Q normalization for numerically stable UCB
  - Policy improvement: action selected proportionally to visit count distribution
  - Integration with PPO-trained ActorCriticGCFNetwork for prior probabilities and value estimates
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
      - parent / action_from_parent: tree structure
      - N: visit count
      - W: cumulative value (sum of backed-up values)
      - Q: mean value W/N
      - P: prior probability from the neural network (for UCB)
      - children: dict mapping action_key → MCTSNode
    """
    __slots__ = ('state', 'parent', 'action_from_parent',
                 'N', 'W', 'Q', 'P', 'children', 'is_terminal')

    def __init__(self, state: np.ndarray, parent: Optional['MCTSNode'] = None,
                 action_from_parent: Optional[np.ndarray] = None, prior: float = 1.0):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.N: int = 0
        self.W: float = 0.0
        self.Q: float = 0.0
        self.P: float = prior       # prior probability from policy network
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
    
    The search loop:
      1. SELECT: traverse tree using UCB-PUCT until a leaf node is reached
      2. EXPAND: generate n_actions candidate actions from the neural policy prior
      3. EVALUATE: use critic to estimate V(leaf) without rollout (zero-latency)
      4. BACKUP: propagate V up through visited nodes
    
    After `num_simulations` iterations, the visit count distribution at the root
    defines the improved policy π̂(a|s_root) used for PPO training targets.
    
    The best single action for inference is argmax(visit_counts).
    """

    def __init__(self, env: AbstractRLEnvironment, network: ActorCriticGCFNetwork,
                 num_simulations: int = 200, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25,
                 n_actions: int = 8):
        """
        Args:
            env: The RL environment (Euler-Mascheroni or generic GCF env)
            network: Trained ActorCriticGCFNetwork
            num_simulations: Number of MCTS rollouts per search call
            c_puct: UCB exploration constant (higher = more exploration)
            dirichlet_alpha: Concentration parameter for root Dirichlet noise
            dirichlet_epsilon: Mixing weight of Dirichlet noise at root
            n_actions: Number of candidate actions to expand at each leaf
        """
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

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Traverse tree selecting max UCB child until a leaf is found."""
        while not node.is_leaf() and not node.is_terminal:
            best_score = -float('inf')
            best_child = None
            for child in node.children.values():
                score = self._ucb_score(node, child)
                if score > best_score:
                    best_score = score
                    best_child = child
            node = best_child
        return node

    @torch.no_grad()
    def _get_policy_value(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Query neural network for action distribution and value estimate.
        Returns sampled actions, their priors, and the value estimate.
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean, std, value = self.network(state_t)
        mean = mean.cpu().numpy().squeeze(0)
        std = std.cpu().numpy().squeeze(0)
        value = value.cpu().item()

        # Sample n_actions candidate actions from the policy distribution
        actions = np.random.normal(mean, std, size=(self.n_actions, len(mean)))
        # Priors as normalized softmax over negative distance from mean (higher prior for closer-to-mean)
        distances = np.linalg.norm(actions - mean, axis=1)
        priors = np.exp(-distances)
        priors /= priors.sum() + 1e-8

        return actions, priors, value

    def _expand(self, node: MCTSNode):
        """Expand a leaf node: generate n_actions children using policy network."""
        actions, priors, value = self._get_policy_value(node.state)

        # Inject Dirichlet noise at root for exploration
        if node.parent is None:
            noise = np.random.dirichlet([self.dirichlet_alpha] * self.n_actions)
            priors = (1 - self.dirichlet_epsilon) * priors + self.dirichlet_epsilon * noise

        for i, (action, prior) in enumerate(zip(actions, priors)):
            child = MCTSNode(state=node.state.copy(), parent=node,
                             action_from_parent=action, prior=float(prior))
            node.children[i] = child

        return value

    def _backup(self, node: MCTSNode, value: float):
        """Backpropagate the value estimate up through the tree."""
        current = node
        while current is not None:
            current.update(value)
            # Update global min/max Q for normalization
            if current.Q < self._q_min:
                self._q_min = current.Q
            if current.Q > self._q_max:
                self._q_max = current.Q
            current = current.parent

    def _simulate_to_leaf(self, node: MCTSNode, initial_state: np.ndarray) -> Tuple[MCTSNode, float]:
        """
        Select a path through the tree, optionally stepping the environment.
        For efficiency, we use the CRITIC for value estimation (no random rollout).
        Returns the reached leaf node and the neural value estimate.
        """
        # Re-run environment to match the node's trajectory
        # For efficiency in the continuous domain, we step the env along the path
        path = []
        current = node
        while current.parent is not None:
            path.append(current.action_from_parent)
            current = current.parent
        path.reverse()

        self.env.reset()
        obs = initial_state.copy()
        for action in path:
            obs, _, done, _ = self.env.step(action)
            if done:
                break

        # The leaf's "state" is now the current env observation
        node.state = obs.copy()

        # Get neural value estimate (no random rollout needed — critic IS the value function)
        _, _, value = self._get_policy_value(obs)
        return node, value

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

        # Build root node
        root = MCTSNode(state=initial_state.copy())

        for _ in range(self.num_simulations):
            # 1. SELECT
            leaf = self._select(root)

            # 2. EXPAND (if not terminal)
            if not leaf.is_terminal:
                self._expand(leaf)
                # After expansion, select the first child for evaluation
                if leaf.children:
                    first_child = list(leaf.children.values())[0]
                    first_child.state = leaf.state.copy()
                    # Quick env step to get child state
                    obs, reward, done, _ = self.env.step(first_child.action_from_parent)
                    first_child.state = obs.copy()
                    first_child.is_terminal = done
                    # 3. EVALUATE via critic
                    _, _, value = self._get_policy_value(obs)
                    # 4. BACKUP
                    self._backup(first_child, value + reward * 0.01)
                else:
                    self._backup(leaf, leaf.Q)
            else:
                self._backup(leaf, 0.0)

        # Extract visit count distribution over root children
        if not root.children:
            # Network not warmed up yet — return random action
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
                               radius_multiplier: float = 3.0) -> Tuple[List, List]:
        """
        High-level API for NeuralMCTSPolyDomain integration.
        Runs MCTS and converts the best continuous action into integer GCF bounds.
        
        Args:
            initial_state: Starting GCF trajectory observation
            original_a_range: Current a_n per-coefficient bounds (list of [min, max])
            original_b_range: Current b_n per-coefficient bounds (list of [min, max])
            radius_multiplier: Scale the action std into a search radius
        
        Returns:
            Updated (a_coef_range, b_coef_range) with tightened bounds
        """
        best_action, _ = self.search(initial_state)
        a_proxy, b_proxy = float(best_action[0]), float(best_action[1])

        # Convert proxy scalar to a search radius around the midpoint
        a_radius = max(int(abs(a_proxy) * radius_multiplier), 2)
        b_radius = max(int(abs(b_proxy) * radius_multiplier), 2)

        new_a_range = []
        for lo, hi in original_a_range:
            mid = (lo + hi) // 2
            new_a_range.append([max(lo, mid - a_radius), min(hi, mid + a_radius)])

        new_b_range = []
        for lo, hi in original_b_range:
            mid = (lo + hi) // 2
            new_b_range.append([max(lo, mid - b_radius), min(hi, mid + b_radius)])

        return new_a_range, new_b_range

    # Type annotation helper
    from typing import List
