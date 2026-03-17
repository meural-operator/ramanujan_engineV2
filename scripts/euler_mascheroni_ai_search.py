import time
from ramanujan.LHSHashTable import LHSHashTable
from ramanujan.enumerators.GPUEfficientGCFEnumerator import GPUEfficientGCFEnumerator
from ramanujan.poly_domains.ContinuousRelaxationDomain import ContinuousRelaxationDomain
from ramanujan.constants import g_const_dict

def main():
    print("Starting AI-Accelerated Baseline Search for Euler-Mascheroni Constant...")
    
    # 1. Create LHSHashTable for Euler-Mascheroni constant
    saved_hash = 'euler_mascheroni.lhs.dept15.db'
    lhs_search_limit = 15
    const_val = g_const_dict['euler-mascheroni']
    
    print("Building left-hand side structure definitions...")
    lhs = LHSHashTable(
        saved_hash,
        lhs_search_limit,
        [const_val]
    )

    # 2. Setup ContinuousRelaxationDomain 
    # We search polynomials of degree 2 for both a_n and b_n.
    # We define a broad initial continuous range and let gradient bounds heavily scale it
    print("Initializing Continuous Relaxation Domain bounding reductions...")
    poly_search_domain = ContinuousRelaxationDomain(
        a_deg=2, a_coef_range=[-15, 15],
        b_deg=2, b_coef_range=[-15, 15],
        lr=0.5, epochs=50
    )
    
    # 3. Setup GPUEfficientGCFEnumerator
    print(f"Deploying GPUEfficientGCFEnumerator with Continuous bounds applied.")
    enumerator = GPUEfficientGCFEnumerator(
        lhs,
        poly_search_domain,
        [const_val]
    )
    
    # 4. Execute
    start_time = time.time()
    results = enumerator.full_execution()
    end_time = time.time()
    
    if len(results) == 0:
        print("No exact GCF conjectures mapped into algebraic LHS values found within iteration bounds.")
    
    enumerator.print_results(results)
    print(f"Total time executed: {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
