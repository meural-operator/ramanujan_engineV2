import math
import numpy as np
from typing import Tuple, Any, Dict
from .AbstractRLEnvironment import AbstractRLEnvironment

class GCFRewardEnvironment(AbstractRLEnvironment):
    """
    Defines the RL environment whose reward function is the numerical convergence rate 
    of the predicted GCF sequence to know mathematical constants.
    """
    def __init__(self, target_value: float, max_steps: int = 50):
        self.target_value = target_value
        self.max_steps = max_steps
        self.current_step = 0
        
        # We start with the base GCF sequence elements
        self.q = 1.0
        self.p = 0.0
        self.prev_q = 0.0
        self.prev_p = 1.0

    def reset(self) -> Tuple[float, float, float, float]:
        self.current_step = 0
        self.q = 1.0
        self.p = 0.0
        self.prev_q = 0.0
        self.prev_p = 1.0
        return (self.prev_q, self.prev_p, self.q, self.p)

    def calculate_reward(self, p_n: float, q_n: float) -> float:
        """
        Reward is strictly proportional to the number of digits matched with the target value.
        """
        if q_n == 0:
            return -10.0 # Heavy penalty for zero division

        predicted_val = p_n / q_n
        error = abs(self.target_value - predicted_val)
        
        if error == 0:
            return 100.0 # Perfect match
            
        # Logarithmic reward based on precision reached
        digits_accurate = -math.log10(error)
        
        # Reward is constrained to realistic digit bounds
        return max(0.0, min(100.0, digits_accurate))

    def step(self, action: Tuple[float, float]) -> Tuple[Tuple[float, float, float, float], float, bool, Dict]:
        """
        RL Step: Applies the chosen polynomial values for the current N depth.
        action: tuple (action_a_n, action_b_n)
        """
        action_a_n, action_b_n = action
        
        next_q = action_a_n * self.q + action_b_n * self.prev_q
        next_p = action_a_n * self.p + action_b_n * self.prev_p
        
        # Periodic scaling
        scale = max(abs(next_q), 1.0)
        next_q /= scale
        next_p /= scale
        
        self.prev_q = self.q / scale
        self.prev_p = self.p / scale
        self.q = next_q
        self.p = next_p
        
        reward = self.calculate_reward(self.p, self.q)
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        obs = (self.prev_q, self.prev_p, self.q, self.p)
        return obs, reward, done, {}
