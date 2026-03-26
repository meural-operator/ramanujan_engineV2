"""
Ramanujan@Home: Local Continuous Discovery Miner (High-Throughput)

This script bypasses the distributed Firebase network to perform
consecutive, immediate brute-force search blocks on known mathematical constants.
It leverages 100% of the local GPU's tensor-core bandwidth.

Supports two operational modes:
  1. NEURAL MCTS MODE: Uses the RL bounding network to intelligently prune the search space
  2. BRUTE FORCE MODE: Pure exhaustive Cartesian product search (wider coverage, slower per target)
"""
import sys
import os
import time

# Ensure the framework root resolves safely 
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from modules.continued_fractions.targets.publishable_targets import PiTarget, ETarget, CatalanTarget
from modules.continued_fractions.engines.GPUEfficientGCFEnumerator import GPUEfficientGCFEnumerator
from modules.continued_fractions.domains.CartesianProductPolyDomain import CartesianProductPolyDomain

USE_NEURAL_MCTS = True  # Uses RL-guided bound pruning via AlphaTensor MCTS

def execute_discovery_run(target_instance, a_degree=2, b_degree=2, bound_radius=15, 
                          mcts_sims=500, use_mcts=False):
    print("\n" + "=" * 70)
    print(f"   TARGET LOCK: Searching for [{target_instance.name.upper()}] Discoveries")
    print("=" * 70)
    
    # 1. Fetch or Auto-Generate the LHS Hash Table
    # This returns a fully initialized LHSHashTable object with bloom filter
    # and correct s_name pointer for disk-backed dictionary access.
    print(f"[1/4] Retrieving or building Mobius transform Hash Table...")
    lhs_obj = target_instance.generate_lhs_hash_table(depth=20)
    print(f"       LHS table ready. Bloom filter loaded. Disk cache: {lhs_obj.s_name}")
    
    # 2. Setup Polynomial Search Domain
    if use_mcts:
        print("[2/4] Running Deep Reinforcement Learning (Neural MCTS) to bound search scope...")
        from modules.continued_fractions.domains.NeuralMCTSPolyDomain import NeuralMCTSPolyDomain
        poly_search_domain = NeuralMCTSPolyDomain(
            a_deg=a_degree, a_coef_range=[-bound_radius, bound_radius],
            b_deg=b_degree, b_coef_range=[-bound_radius, bound_radius],
            target_val=target_instance._val,
            mcts_simulations=mcts_sims  
        )
    else:
        print("[2/4] Using pure brute-force Cartesian product (full combinatorial coverage)...")
        poly_search_domain = CartesianProductPolyDomain(
            a_deg=a_degree, a_coef_range=[-bound_radius, bound_radius],
            b_deg=b_degree, b_coef_range=[-bound_radius, bound_radius]
        )
    
    print(f"       a_n polynomial bounds: {poly_search_domain.a_coef_range}")
    print(f"       b_n polynomial bounds: {poly_search_domain.b_coef_range}")
    
    an_count = poly_search_domain.get_an_length()
    bn_count = poly_search_domain.get_bn_length()
    total_evals = an_count * bn_count
    print(f"       Tensor Iteration Space: {an_count:,} x {bn_count:,} = {total_evals:,} GPU targeted GCF evaluations")
    
    # 3. Deploy GPU enumerator with the properly initialized LHSHashTable object
    print("\n[3/4] Initializing GPU-accelerated GCF enumerator...")
    enumerator = GPUEfficientGCFEnumerator(
        lhs_obj,
        poly_search_domain,
        [target_instance._val]
    )

    # 4. Execute full search
    print(f"[4/4] Executing full enumeration. Monitor GPU usage with nvidia-smi.")
    start_time = time.time()
    results = enumerator.full_execution()
    elapsed = time.time() - start_time
    
    # 5. Output Summary
    print("\n" + "-" * 50)
    print(f"   RESULTS FOR {target_instance.name.upper()}")
    print("-" * 50)
    if len(results) == 0:
        print("  No conjectures found within current search bounds.")
    else:
        print(f"  Found {len(results)} conjecture(s)!")
        enumerator.print_results(results)
    
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"  Target runtime: {int(hours)}h {int(minutes)}m {seconds:.1f}s\n")
    return results

def main():
    print("=========================================================================")
    print("   Ramanujan@Home: Local Continuous Discovery Miner (High-Throughput)    ")
    print("=========================================================================\n")
    print(" This script bypasses the distributed Firebase network to perform")
    print(" consecutive, immediate brute-force search blocks on known constants.")
    print(" It leverages 100% of your local GPU's tensor-core boundaries.")
    if USE_NEURAL_MCTS:
        print(" Mode: NEURAL MCTS (RL-guided bound pruning)")
    else:
        print(" Mode: BRUTE FORCE (full Cartesian product enumeration)")
    
    # Configure the generic mathematically publishable targets
    targets = [
        CatalanTarget(),
        ETarget(),
        PiTarget()
    ]
    
    # Let the miner cycle through them sequentially
    total_start = time.time()
    all_discoveries = 0
    
    for target in targets:
        # Degree 2 with radius 25 gives (50+1)^3 = 132,651 per polynomial side
        # Total: ~17.6 billion combinations (will be chunked by GPU VRAM)
        # For faster initial runs, use smaller radius (e.g., 10-15)
        found = execute_discovery_run(
            target_instance=target, 
            a_degree=2, 
            b_degree=2, 
            bound_radius=25, 
            mcts_sims=500,
            use_mcts=USE_NEURAL_MCTS
        )
        all_discoveries += len(found)
        
    total_elapsed = time.time() - total_start
    h, r = divmod(total_elapsed, 3600)
    m, s = divmod(r, 60)
    
    print("=========================================================================")
    print("                            MINER COMPLETE")
    print("=========================================================================")
    print(f" Total Global Found : {all_discoveries} mathematical formulas")
    print(f" Total Global Time  : {int(h)}h {int(m)}m {s:.1f}s")
    print("=========================================================================")

if __name__ == '__main__':
    main()
