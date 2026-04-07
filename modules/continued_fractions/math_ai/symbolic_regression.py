"""
Symbolic Regression for GCF Sequence Discovery.

Uses PySR to discover closed-form analytical expressions for a(n) and b(n) 
sequences from *actual* discovered GCF coefficient data, not synthetic curves.
"""
import numpy as np
try:
    from pysr import PySRRegressor
except ImportError:
    PySRRegressor = None

class ParamSymbolicRegressor:
    """
    Integrates PySR to perform symbolic regression on successful generating sequences.
    Instead of manually crafting polynomial domains, PySR discovers
    analytical functional forms for a(n) and b(n) from real GCF data.
    """
    def __init__(self, iterations=40):
        if PySRRegressor is None:
            raise ImportError("PySR is not installed. Please install it to use symbolic regression.")
            
        self.model = PySRRegressor(
            niterations=iterations,
            binary_operators=["+", "*", "-", "/"],
            unary_operators=["square", "cube", "exp", "inv(x) = 1/x"],
            loss="loss(prediction, target) = (prediction - target)^2",
            maxsize=15,
        )

    def fit_sequence(self, n_indices: np.ndarray, sequence_values: np.ndarray):
        """
        Fits a functional representation P(n) to an observed numerical sequence 
        that yields a high convergence metric.
        
        :param n_indices: Array of integer indices [1, 2, 3, ...]
        :param sequence_values: Array of target values the sequence produced [a_1, a_2, ...]
        """
        X = n_indices.reshape(-1, 1)
        self.model.fit(X, sequence_values)
        return self.model.sympy()

    def discover_from_gcf_hit(self, a_values: list, b_values: list):
        """
        Given actual a(n) and b(n) coefficient sequences extracted from a confirmed
        GCF discovery, use PySR to find closed-form symbolic expressions for each.
        
        This is the correct way to use symbolic regression in the Ramanujan Machine:
        the GPU sweep finds raw integer coefficient sequences that produce convergent
        GCFs, and this method extracts the underlying mathematical pattern.
        
        Args:
            a_values: List of actual a(n) values from a discovered GCF [a(0), a(1), ...]
            b_values: List of actual b(n) values from a discovered GCF [b(0), b(1), ...]
            
        Returns:
            Tuple of (a_n_symbolic, b_n_symbolic) sympy expressions, or None if fitting fails.
        """
        if not a_values or not b_values:
            raise ValueError("Must provide non-empty a(n) and b(n) sequences from a real GCF hit.")
        
        n_a = np.arange(len(a_values))
        n_b = np.arange(len(b_values))
        
        a_sym = self.fit_sequence(n_a, np.array(a_values, dtype=np.float64))
        
        # Create a fresh model for b(n) to avoid interference
        b_model = PySRRegressor(
            niterations=self.model.niterations,
            binary_operators=["+", "*", "-", "/"],
            unary_operators=["square", "cube", "exp", "inv(x) = 1/x"],
            loss="loss(prediction, target) = (prediction - target)^2",
            maxsize=15,
        )
        b_model.fit(n_b.reshape(-1, 1), np.array(b_values, dtype=np.float64))
        b_sym = b_model.sympy()
        
        return a_sym, b_sym

    def discover_domain(self, a_values: list, b_values: list, max_depth: int = 50):
        """
        Given actual coefficient sequences from a discovered GCF, find symbolic forms.
        
        This replaces the previous implementation which fabricated a synthetic curve
        c*(1-1/n) — that curve had no mathematical relationship to actual GCF 
        recurrences and produced meaningless symbolic fits.
        
        Args:
            a_values: List of actual a(n) coefficient values from a confirmed GCF hit.
            b_values: List of actual b(n) coefficient values from a confirmed GCF hit.
            max_depth: Maximum number of terms to use for fitting (truncates if longer).
            
        Returns:
            Tuple of (a_n_expr, b_n_expr) as sympy expressions.
        """
        a_vals = a_values[:max_depth]
        b_vals = b_values[:max_depth]
        return self.discover_from_gcf_hit(a_vals, b_vals)
