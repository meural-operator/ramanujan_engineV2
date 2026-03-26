from typing import List, Tuple, Dict
from core.interfaces.base_engine import ExecutionEngine
from core.interfaces.base_problem import TargetProblem
from modules.continued_fractions.domains.CartesianProductPolyDomain import CartesianProductPolyDomain
from modules.continued_fractions.LHSHashTable import LHSHashTable

from modules.continued_fractions.engines.GPUEfficientGCFEnumerator import GPUEfficientGCFEnumerator

class CUDAEnumerator(ExecutionEngine):
    """
    V3 modular adapter binding the highly-optimized V2 PyTorch Matrix Exhaust loop
    into the generic plug-and-play Bounding Pipeline.
    """
    @property
    def engine_id(self) -> str:
        return "CUDA-TensorCore"
        
    def batch_evaluate(self, a_bounds: List[List[int]], b_bounds: List[List[int]], target: TargetProblem) -> List[Dict]:
        a_deg = len(a_bounds) - 1
        b_deg = len(b_bounds) - 1
        
        domain = CartesianProductPolyDomain(
            a_deg=a_deg, a_coef_range=[0, 0],
            b_deg=b_deg, b_coef_range=[0, 0]
        )
        domain.a_coef_range = a_bounds
        domain.b_coef_range = b_bounds
        domain._setup_metadata()
        
        # Load arbitrary generalized LHS keys from the TargetProblem implementation
        # generate_lhs_hash_table now returns a fully initialized LHSHashTable object
        lhs_obj = target.generate_lhs_hash_table(depth=30)
        
        try:
            target_val = target._val
        except AttributeError:
            target_val = 0.577215  # generic fallback
        
        # If generate_lhs_hash_table already returns an LHSHashTable, use it directly.
        # Otherwise, wrap it for backward compatibility with targets that return raw dicts.
        if isinstance(lhs_obj, LHSHashTable):
            legacy_lhs = lhs_obj
        else:
            # Backward compatibility: wrap raw dict in LHSHashTable shell
            legacy_lhs = LHSHashTable("", 30, [target_val])
            legacy_lhs.lhs_possibilities = lhs_obj
            
        enumerator = GPUEfficientGCFEnumerator(
            legacy_lhs,
            domain,
            [target_val]
        )
        
        # Fire V2 CUDA Sweep Thread
        raw_hits = enumerator.full_execution(verbose=True)
        
        # Format explicitly mathematically for Coordinator transmission
        results = []
        for hit in raw_hits:
            results.append({
                "lhs_key": hit.lhs_key,
                "a_coef": hit.rhs_an_poly,
                "b_coef": hit.rhs_bn_poly
            })
            
        return results
