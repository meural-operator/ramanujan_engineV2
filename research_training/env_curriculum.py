import os
import sys

# Ensure repo root is available
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ramanujan.math_ai.environments.EulerMascheroniEnvironment import EulerMascheroniEnvironment
import numpy as np
from typing import Tuple, Dict, Any

class CurriculumEulerMascheroniEnv(EulerMascheroniEnvironment):
    """
    Wrapper around the Euler-Mascheroni Environment that supports Curriculum Learning.
    
    Instead of forcing the agent to predict 150 valid GCF depth steps immediately 
    (which generates massive negative gradients due to overflow and 0 rewards), 
    this wrapper starts at `max_steps=20` and only increments the difficulty when 
    the trainer signals that a mastery threshold has been met.
    """
    
    def __init__(self, config: dict):
        self.config = config['environment']
        
        # Initialize at easiest difficulty
        self.current_max_steps = self.config['max_steps_initial']
        
        # Upper bound limitation
        self.absolute_limit = self.config['max_steps_limit']
        
        super().__init__(max_steps=self.current_max_steps)
        
        # State tracking for the curriculum manager
        self.consecutive_mastery_episodes = 0
        self.total_episodes_at_level = 0
        self.patience = self.config['curriculum_patience_eps']
        self.threshold = self.config['curriculum_promotion_reward']
        
    def check_promotion(self, recent_mean_reward: float) -> bool:
        """
        Called by the training loop every N episodes.
        If the agent has mastered this depth, we deepen the GCF requirement.
        Returns True if a promotion occurred, False otherwise.
        """
        self.total_episodes_at_level += 1
        
        # Cannot exceed absolute limit
        if self.current_max_steps >= self.absolute_limit:
            return False
            
        # Check mastery Condition
        if recent_mean_reward >= self.threshold:
            self.consecutive_mastery_episodes += 1
        else:
            self.consecutive_mastery_episodes = 0
            
        # If sustained mastery, promote
        if self.consecutive_mastery_episodes >= self.patience:
            old_steps = self.current_max_steps
            new_steps = min(self.absolute_limit, old_steps + self.config['curriculum_step_increase'])
            
            self.current_max_steps = new_steps
            self.max_steps = new_steps # Update base class
            
            # Reset tracking for new level
            self.consecutive_mastery_episodes = 0
            self.total_episodes_at_level = 0
            
            print(f"\n[🎓 Curriculum Promotion] Mastered depth {old_steps}! Environment difficulty increased to max_steps={new_steps}\n")
            return True
            
        return False
        
    def reset(self) -> np.ndarray:
        return super().reset()
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        # Optional: We can shape the reward here if we want depth-relative normalization,
        # but the base `calculate_reward` measuring decimal precision is already scale-invariant.
        return super().step(action)
