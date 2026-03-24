"""
Euler-Mascheroni Specific RL Environment.

The Euler-Mascheroni constant γ ≈ 0.5772156649015328...
is one of the deepest unsolved constants in mathematics.
No simple GCF formula with polynomial coefficients has been proven to converge to γ.

This environment shapes the reward to:
  1. Reward digit-accuracy (primary signal)
  2. Bonus for *improving* convergence rate step-over-step (shaped exploration reward)
  3. Numerical overflow guard (early termination with penalty)
  4. Normalized state observation for stable neural network input
"""
import math
import mpmath
import numpy as np
from typing import Tuple, Dict, Any

from .AbstractRLEnvironment import AbstractRLEnvironment

# High-precision Euler-Mascheroni constant (mpmath's reference value)
_GAMMA = float(mpmath.euler)  # 0.5772156649015328606...


class EulerMascheroniEnvironment(AbstractRLEnvironment):
    """
    Reinforcement learning environment for discovering GCF representations of γ.
    
    State:
        A 4-tuple of normalized GCF numerator/denominator evolution:
        [sign(prev_q)*log1p(|prev_q|), sign(prev_p)*log1p(|prev_p|),
         sign(q)*log1p(|q|),           sign(p)*log1p(|p|)]
        Log-scaling prevents float overflow while preserving magnitude ordering.
    
    Action:
        Continuous 2D vector (a_n_proxy, b_n_proxy) representing the polynomial
        coefficient scaling factors for the current GCF depth.
    
    Reward:
        - Primary: digits of precision gained (log10 scale) vs target γ
        - Bonus: +2 for each digit of improvement over the previous best precision
        - Penalty: -20 for numerical overflow (|p| or |q| > 1e12 before normalization)
        - Penalty: -10 for q = 0 division
    """

    env_name: str = "euler_mascheroni"
    target_val: float = _GAMMA

    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps
        self.target_value = _GAMMA
        self.reset()

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.prev_q = 0.0
        self.prev_p = 1.0
        self.q = 1.0
        self.p = _GAMMA  # warm-start: first GCF term is γ itself
        self.best_digits = 0.0
        return self._get_obs()

    def _safe_log_norm(self, x: float) -> float:
        """Sign-preserving log1p normalization that handles extreme values."""
        return math.copysign(math.log1p(abs(x)), x)

    def _get_obs(self) -> np.ndarray:
        return np.array([
            self._safe_log_norm(self.prev_q),
            self._safe_log_norm(self.prev_p),
            self._safe_log_norm(self.q),
            self._safe_log_norm(self.p),
        ], dtype=np.float32)

    def calculate_reward(self, p: float, q: float) -> float:
        """Primary reward: log10(1 / |γ - p/q|), i.e. digits of precision."""
        if abs(q) < 1e-15:
            return -10.0

        predicted = p / q
        error = abs(self.target_value - predicted)

        if error == 0.0:
            return 100.0

        digits = -math.log10(error + 1e-300)
        return max(0.0, min(100.0, digits))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Apply one GCF recurrence step using the action as (a_n, b_n) proxies.
        
        The action is treated as a *scaling multiplier* applied to the running
        convergent before update — this lets the agent learn to guide the GCF
        trajectory direction without prescribing exact integer coefficients.
        """
        a_n_proxy, b_n_proxy = float(action[0]), float(action[1])

        # Overflow guard before update
        if abs(self.q) > 1e12 or abs(self.p) > 1e12:
            return self._get_obs(), -20.0, True, {"overflow": True}

        # GCF recurrence: q_n = a_n * q_{n-1} + b_n * q_{n-2}
        next_q = a_n_proxy * self.q + b_n_proxy * self.prev_q
        next_p = a_n_proxy * self.p + b_n_proxy * self.prev_p

        # Periodic scaling to keep magnitudes tractable
        scale = max(abs(next_q), 1.0)
        next_q /= scale
        next_p /= scale
        self.prev_q = self.q / scale
        self.prev_p = self.p / scale
        self.q = next_q
        self.p = next_p

        # Compute primary reward
        primary_reward = self.calculate_reward(self.p, self.q)

        # Convergence-rate bonus: extra reward for improving our best digit count
        improvement_bonus = 0.0
        if primary_reward > self.best_digits:
            improvement_bonus = 2.0 * (primary_reward - self.best_digits)
            self.best_digits = primary_reward

        reward = primary_reward + improvement_bonus

        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self._get_obs(), reward, done, {
            "digits_accurate": primary_reward,
            "improvement_bonus": improvement_bonus,
        }
