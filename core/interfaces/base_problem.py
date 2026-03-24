from abc import ABC, abstractmethod

class TargetProblem(ABC):
    """
    Abstract base class defining a mathematical constant targeted for discovery.
    Enforces required methods for precision control and formula verification.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """The canonical string designation of the target constant (e.g., 'euler-mascheroni')."""
        pass

    @property
    @abstractmethod
    def precision(self) -> int:
        """The decimal/digit precision strictly required for rigorous verification."""
        pass

    @abstractmethod
    def generate_lhs_hash_table(self, depth: int) -> dict:
        """
        Generates the meet-in-the-middle structural hash table caching the partial
        evaluations of Left-Hand Side equations (Used to bypass brute-force on identical branches).
        """
        pass

    @abstractmethod
    def verify_match(self, a_coef: tuple, b_coef: tuple) -> float:
        """
        Rigorously evaluates the derived polynomial coefficient structure against the 
        target constant up to the defined property precision scale.
        
        Returns:
            Absolute mathematical error (float). Discoveries typically mandate error < 1e-100.
        """
        pass
