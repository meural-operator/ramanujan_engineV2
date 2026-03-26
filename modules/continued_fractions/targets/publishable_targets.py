import os
import mpmath
try:
    import dill as pickle
except ImportError:
    import pickle
from core.interfaces.base_problem import TargetProblem
from modules.continued_fractions.utils.mobius import EfficientGCF

g_N_verify_terms = 1000
g_N_verify_compare_length = 100

class AbstractConstantTarget(TargetProblem):
    def __init__(self, name: str, val, db_filename: str):
        self._name = name
        self._val = val
        self._db_filename = db_filename

    @property
    def name(self) -> str:
        return self._name

    @property
    def precision(self) -> int:
        return g_N_verify_compare_length

    def generate_lhs_hash_table(self, depth: int):
        """
        Returns a fully initialized LHSHashTable object.
        If the .db file already exists on disk, the constructor loads it natively.
        If not, the constructor generates the enumeration and writes it to disk.
        Either way, the returned object is a valid LHSHashTable ready for GPU matching.
        """
        from modules.continued_fractions.LHSHashTable import LHSHashTable
        
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        db_path = os.path.join(repo_root, self._db_filename)
        
        if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
            print(f"[*] Missing or corrupt LHS Database for [{self._name}].")
            print(f"[*] Dynamically generating {self._db_filename} (Depth: {depth}). This may take ~10-30 seconds...")
            # Clean up corrupt 0-byte files
            if os.path.exists(db_path) and os.path.getsize(db_path) == 0:
                os.remove(db_path)
        else:
            print(f"[*] Loading pre-built LHS Database: {self._db_filename}")
            
        # LHSHashTable constructor handles both generation and loading natively.
        # It writes the dict to <name>.db and then sets self.lhs_possibilities = None
        # to free memory. The object retains the bloom filter for fast 'in' checks,
        # and will re-load the dict from disk on demand via _get_by_key().
        lhs = LHSHashTable(
            name=db_path, 
            search_range=depth, 
            const_vals=[self._val]
        )
        return lhs

    def verify_match(self, a_coef: tuple, b_coef: tuple) -> float:
        """
        Dynamically extracts the strict absolute mathematical error.
        Designed to gracefully support polynomials of ANY mathematical degree 
        rather than just hardcoded quadratics.
        """
        with mpmath.workdps(g_N_verify_terms * 2):
            an = self._create_an_series(a_coef, g_N_verify_terms)
            bn = self._create_bn_series(b_coef, g_N_verify_terms)
            gcf = EfficientGCF(an, bn)
            
            try:
                val = gcf.evaluate()
                if mpmath.isinf(val) or mpmath.isnan(val):
                    return float('inf')
                
                error = abs(val - self._val)
                return float(error)
            except Exception:
                return float('inf')

    def _create_an_series(self, poly_coefs, n_terms):
        """
        Supports list of coefficients [c_d, c_{d-1}, ..., c_1, c_0] for any degree.
        """
        degree = len(poly_coefs) - 1
        return [0] + [
            sum(c * (i ** (degree - k)) for k, c in enumerate(poly_coefs))
            for i in range(1, n_terms)
        ]

    def _create_bn_series(self, poly_coefs, n_terms):
        degree = len(poly_coefs) - 1
        return [0] + [
            sum(c * (i ** (degree - k)) for k, c in enumerate(poly_coefs))
            for i in range(1, n_terms)
        ]

class PiTarget(AbstractConstantTarget):
    def __init__(self):
        super().__init__("pi", mpmath.pi, "pi.db")

class ETarget(AbstractConstantTarget):
    def __init__(self):
        super().__init__("e", mpmath.e, "e.db")

class CatalanTarget(AbstractConstantTarget):
    def __init__(self):
        super().__init__("catalan", mpmath.catalan, "catalan.db")

class GoldenRatioTarget(AbstractConstantTarget):
    def __init__(self):
        super().__init__("golden_ratio", mpmath.phi, "golden_ratio.db")
