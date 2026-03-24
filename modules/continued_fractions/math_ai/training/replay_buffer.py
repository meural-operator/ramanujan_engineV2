"""
Trajectory Replay Buffer with GAE (Generalized Advantage Estimation).
Research-grade rollout storage for PPO training of the GCF discovery agent.
"""
import numpy as np
import torch
from typing import List, Tuple


class TrajectoryBuffer:
    """
    Stores full rollout trajectories for one PPO update cycle.
    
    Each call to store() saves one (state, action, reward, value, log_prob, done) tuple.
    After collecting enough steps, compute_gae() calculates the TD-lambda advantages
    and value targets, then the buffer can be sampled in random mini-batches for PPO updates.
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95,
                 device: torch.device = None):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device or torch.device('cpu')
        self._reset()

    def _reset(self):
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []
        self.advantages: torch.Tensor = None
        self.returns: torch.Tensor = None

    def store(self, state: np.ndarray, action: np.ndarray, reward: float,
              value: float, log_prob: float, done: bool):
        self.states.append(state.astype(np.float32))
        self.actions.append(action.astype(np.float32))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.dones.append(bool(done))

    def compute_gae(self, last_value: float = 0.0):
        """
        Computes GAE advantages and discounted returns in-place.
        Call this after a complete rollout, before calling get_batches().
        
        Args:
            last_value: Bootstrap value for the state after the final stored step.
                        Set to 0.0 if the episode ended naturally.
        """
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            next_value = self.values[t + 1] if t < T - 1 else last_value
            next_non_terminal = 0.0 if self.dones[t] else 1.0

            delta = (self.rewards[t]
                     + self.gamma * next_value * next_non_terminal
                     - self.values[t])
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)

        # Normalize advantages for stable learning
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        self.advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        self.returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

    def get_batches(self, batch_size: int) -> List[Tuple[torch.Tensor, ...]]:
        """
        Returns shuffled mini-batches of (states, actions, log_probs, advantages, returns).
        Must call compute_gae() first.
        """
        assert self.advantages is not None, "Call compute_gae() before get_batches()"
        T = len(self.states)
        indices = np.random.permutation(T)

        states_t = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(np.array(self.actions), dtype=torch.float32, device=self.device)
        log_probs_t = torch.tensor(self.log_probs, dtype=torch.float32, device=self.device)

        batches = []
        for start in range(0, T, batch_size):
            idx = indices[start: start + batch_size]
            batches.append((
                states_t[idx],
                actions_t[idx],
                log_probs_t[idx],
                self.advantages[idx],
                self.returns[idx],
            ))
        return batches

    def clear(self):
        self._reset()

    def __len__(self):
        return len(self.rewards)
