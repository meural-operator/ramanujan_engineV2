import os
import sys

try:
    from ramanujan.LHSHashTable import LHSHashTable
except ModuleNotFoundError:
    # Local Developer Fallback: Instantly resolves the library if pip installation is bypassed.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'RamanujanMachine'))
    sys.path.append(repo_root)
    from ramanujan.LHSHashTable import LHSHashTable

from ramanujan.constants import g_const_dict
from ramanujan.poly_domains.CartesianProductPolyDomain import CartesianProductPolyDomain
from ramanujan.enumerators.GPUEfficientGCFEnumerator import GPUEfficientGCFEnumerator

class RamanujanExecutor:
    """
    Acts as the bridge between the lightweight client REST payloads
    and the heavyweight PyTorch multi-threaded V2 execution engine.
    """
    def __init__(self):
        # Assumes the user has the generated DB locally or it generates it automatically.
        # Generating a depth 30 DB on the fly takes about ~5 seconds.
        self.lhs_db_path = "euler_mascheroni.db"
        self.const_name = "euler-mascheroni"
        self.const_val = g_const_dict[self.const_name]
        
    def execute_work_unit(self, work_unit):
        print(f"\n==================================================")
        print(f"[*] Starting Engine V2 for Work Unit #{work_unit['id']}")
        print(f"[*] Target Constant: {work_unit['constant_name']}")
        print(f"[*] Execution Grid: {work_unit['range']}")
        print(f"==================================================\n")
        
        # Translate the server JSON bounds strictly
        min_bound, max_bound = work_unit['range']
        a_deg = work_unit['a_deg']
        b_deg = work_unit['b_deg']
        
        # 1. Initialize the LHS Hash Table boundary logic
        # If the file exists, it loads instantly. If not, it builds in 5 seconds.
        lhs = LHSHashTable(
            self.lhs_db_path,
            30,  # Depth limit
            [self.const_val]
        )
        
        # 2. Construct the bounded PolyDomain exclusively for this Node's partition
        # We use Cartesian bounds natively here because the Central Server is the one executing
        # the Neural MCTS macro-logic to determine the most promising partition bounds.
        poly_search_domain = CartesianProductPolyDomain(
            a_deg=a_deg, a_coef_range=[min_bound, max_bound],
            b_deg=b_deg, b_coef_range=[min_bound, max_bound]
        )
        
        # 3. Spin up the advanced Async dual-thread Engine V2 Enumerator
        enumerator = GPUEfficientGCFEnumerator(
            lhs,
            poly_search_domain,
            [self.const_val]
        )
        
        # Deploy execution. The GPU will rip through the specific bounds matrix,
        # dropping preliminary hits into the thread-safe Queue where the CPU will validate using mpmath.
        verified_hits = enumerator.full_execution(verbose=True)
        
        print(f"\n[+] Work Unit #{work_unit['id']} computation finished. Found {len(verified_hits)} ultra-verified hits.")
        return verified_hits
