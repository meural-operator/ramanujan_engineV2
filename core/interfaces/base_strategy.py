from abc import ABC, abstractmethod
from typing import Tuple, List

class BoundingStrategy(ABC):
    """
    Abstract base class defining a mathematical or machine learning heuristic
    for stripping topological mass from polynomial bounding grids prior to evaluation.
    """
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """A canonical string identifier for logging telemetry (e.g., 'mcts_alpha_tensor')."""
        pass

    @abstractmethod
    def prune_bounds(self, raw_a_bounds: List[List[int]], raw_b_bounds: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Receives raw combinatorial bounding constraints and mathematically/heuristically 
        narrows them to eliminate barren execution structures.
        
        Args:
            raw_a_bounds: Nested list of [min, max] ranges for numerator coefficients.
            raw_b_bounds: Nested list of [min, max] ranges for denominator coefficients.
            
        Returns:
            Tuple containing the refined (tightened) a_bounds and b_bounds.
        """
        pass
