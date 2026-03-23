import mpmath
from ramanujan.interfaces.base_constant import TargetConstant
from ramanujan.utils.mobius import EfficientGCF

g_N_verify_terms = 1000
g_N_verify_compare_length = 100

class EulerMascheroniTarget(TargetConstant):
    def __init__(self):
        self._val = mpmath.euler

    @property
    def name(self) -> str:
        return "euler-mascheroni"

    @property
    def precision(self) -> int:
        return g_N_verify_compare_length

    def generate_lhs_hash_table(self, depth: int) -> dict:
        """
        Legacy EM evaluation loads pre-computed LHS from euler_mascheroni.db.
        This provides the structural hook for future dynamic generation.
        """
        import pickle
        import os
        
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        db_path = os.path.join(repo_root, "euler_mascheroni.db")
        
        if os.path.exists(db_path):
            with open(db_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(
                f"Missing {db_path}. Please run `scripts/seed_euler_mascheroni_db.py` "
                "first to pre-calculate the mathematical LHS structures before running the compute node!"
            )
        return {}

    def verify_match(self, a_coef: tuple, b_coef: tuple) -> float:
        """
        Recreates the legacy _cpu_verify_worker computation, but returns the 
        strict absolute mathematical error instead of a string comparison to abstract it globally.
        """
        with mpmath.workdps(g_N_verify_terms * 2):
            an = self._create_an_series(a_coef, g_N_verify_terms)
            bn = self._create_bn_series(b_coef, g_N_verify_terms)
            gcf = EfficientGCF(an, bn)
            
            try:
                val = gcf.evaluate()
                if mpmath.isinf(val) or mpmath.isnan(val):
                    return float('inf')
                
                # Check distance from the global standard target
                error = abs(val - self._val)
                return float(error)
            except Exception:
                return float('inf')

    def _create_an_series(self, poly_coefs, n_terms):
        a2, a1, a0 = poly_coefs
        return [0] + [a2*(i**2) + a1*i + a0 for i in range(1, n_terms)]

    def _create_bn_series(self, poly_coefs, n_terms):
        b2, b1, b0 = poly_coefs
        return [0] + [b2*(i**2) + b1*i + b0 for i in range(1, n_terms)]
