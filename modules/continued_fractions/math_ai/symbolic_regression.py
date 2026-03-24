import numpy as np
try:
    from pysr import PySRRegressor
except ImportError:
    PySRRegressor = None

class RamanujanSymbolicRegressor:
    """
    Integrates PySR to perform symbolic regression on successful generating sequences.
    Instead of manually crafting polynomial domains, PySR discovers
    analytical functional forms that maximize the discovery of convergent GCFs.
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

    def discover_domain(self, target_constant_vals: list, max_depth: int = 50):
        """
        Given a target mathematical constant, attempt to build a functional
        approximation of the sequence convergents using PySR.
        We generate a pseudo-sequence converging to one of the constants 
        and fit the PySR model to extract a symbolic form.
        """
        if not target_constant_vals:
            raise ValueError("Must provide at least one target constant.")

        # Generates a pseudo-sequence converging asymptotically to the target
        n_indices = np.arange(1, max_depth + 1)
        synthetic_trace = np.array([float(target_constant_vals[0]) * (1.0 - 1.0/n) for n in n_indices])
        
        return self.fit_sequence(n_indices, synthetic_trace)
