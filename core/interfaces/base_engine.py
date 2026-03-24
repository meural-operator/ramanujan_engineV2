from abc import ABC, abstractmethod
from typing import List, Tuple
from core.interfaces.base_problem import TargetProblem

class ExecutionEngine(ABC):
    """
    Abstract base class defining the high-performance execution backbone (CPU/GPU) 
    that physically computes generalized fractions.
    """
    @property
    @abstractmethod
    def engine_id(self) -> str:
        """Identifier distinguishing the compute platform (e.g., 'CUDA-Ada', 'CPU-Multiproc')."""
        pass

    @abstractmethod
    def batch_evaluate(self, 
                       a_bounds: List[List[int]], 
                       b_bounds: List[List[int]], 
                       target: TargetProblem) -> List[Tuple]:
        """
        Conducts parallel combinatorial generation and mass evaluation of generalized
        continued fractions across the defined bounding subspace.
        
        Args:
            a_bounds: Final tightened numerator combinatorial limits.
            b_bounds: Final tightened denominator combinatorial limits.
            target: The TargetProblem module injection prescribing the strict mathematical rules.
            
        Returns:
            List of rigorous mathematically verified matches formatted as Tuples of coefficient arrays.
        """
        pass
