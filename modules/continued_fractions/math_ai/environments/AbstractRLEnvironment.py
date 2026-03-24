from abc import ABC, abstractmethod
from typing import Tuple, Any

class AbstractRLEnvironment(ABC):
    """
    Generalized abstract interface for Reinforcement Learning mathematical
    environments. Designed to emulate an OpenAI Gym `env` structure, making
    it plug-and-play for different discovery agents (DQN, PPO, Neural MCTS).
    """
    
    @abstractmethod
    def reset(self) -> Any:
        """
        Resets the environment state to begin a new episode/sequence generation.
        Returns the initial observation state.
        """
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        """
        Executes one time step within the environment based on the given action.
        
        Args:
            action: The action selected by the agent (e.g., coefficient values).
            
        Returns:
            observation (Any): The agent's observation of the current environment.
            reward (float): The amount of reward returned after previous action.
            done (bool): Whether the episode has ended (e.g., max depth reached).
            info (dict): Contains auxiliary diagnostic information.
        """
        pass
    
    @abstractmethod
    def calculate_reward(self, current_state: Any) -> float:
        """
        Specific mathematical logic defining the reward landscape.
        """
        pass
