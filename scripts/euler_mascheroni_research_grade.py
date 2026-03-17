import time
from ramanujan.LHSHashTable import LHSHashTable
from ramanujan.enumerators.GPUEfficientGCFEnumerator import GPUEfficientGCFEnumerator
from ramanujan.poly_domains.ContinuousRelaxationDomain import ContinuousRelaxationDomain
from ramanujan.constants import g_const_dict

def main():
    print("=" * 70)
    print("   RESEARCH GRADE: Euler-Mascheroni GCF Conjecture Discovery")
    print("=" * 70)
    print()
    
    # ───────────────────────────────────────────────────────────────────
    # SEARCH SPACE SIZING GUIDE (degree d, range [-R, R]):
    #   Combinations per side = (2R+1)^(d+1)
    #   Total GCF evaluations = combinations_a × combinations_b
    #
    #   deg=2, R=30  → 61^3 ≈ 227K per side → ~51B total   (~hours)
    #   deg=3, R=10  → 21^4 ≈ 194K per side → ~38B total   (~hours)
    #   deg=3, R=15  → 31^4 ≈ 923K per side → ~852B total  (~days)
    #   deg=3, R=20  → 41^4 ≈ 2.8M per side → ~8T total    (~weeks)
    #
    # Current config: deg=3, R=20 with gentle pruning → multi-hour run.
    # The gradient descent will prune the 2.8M down significantly.
    # ───────────────────────────────────────────────────────────────────
    
    # 1. Build LHS hash table for Euler-Mascheroni (γ ≈ 0.5772...)
    print("[1/4] Building Mobius transform hash table for γ (depth=30)...")
    saved_hash = 'euler_mascheroni.lhs.dept30.db'
    lhs_search_limit = 30
    const_val = g_const_dict['euler-mascheroni']
    
    lhs = LHSHashTable(
        saved_hash,
        lhs_search_limit,
        [const_val]
    )
    print(f"       LHS table ready.\n")
    
    # 2. ContinuousRelaxation prunes bounds via gradient descent
    #    Using GENTLE pruning (low lr, fewer epochs) so we don't over-prune
    print("[2/4] Running PyTorch gradient descent to prune coefficient bounds...")
    print("       Initial range: [-20, 20] per coefficient, degree 3 (4 coefficients)")
    poly_search_domain = ContinuousRelaxationDomain(
        a_deg=3, a_coef_range=[-20, 20],
        b_deg=3, b_coef_range=[-20, 20],
        lr=0.01, epochs=50  # Gentle: don't over-prune
    )
    
    # Report optimized bounds
    a_widths = [r[1] - r[0] for r in poly_search_domain.a_coef_range]
    b_widths = [r[1] - r[0] for r in poly_search_domain.b_coef_range]
    print(f"       Optimized a_n bounds: {poly_search_domain.a_coef_range}")
    print(f"       Optimized b_n bounds: {poly_search_domain.b_coef_range}")
    an_count = poly_search_domain.get_an_length()
    bn_count = poly_search_domain.get_bn_length()
    total_evals = an_count * bn_count
    print(f"       Search space: {an_count:,} × {bn_count:,} = {total_evals:,} GCF evaluations\n")
    
    # 3. Deploy GPU enumerator
    print("[3/4] Initializing GPU-accelerated GCF enumerator...")
    enumerator = GPUEfficientGCFEnumerator(
        lhs,
        poly_search_domain,
        [const_val]
    )

    # 4. Execute full search
    print(f"\n[4/4] Executing full enumeration...")
    print(f"       This may take several hours. Monitor GPU usage with nvidia-smi.\n")
    start_time = time.time()
    results = enumerator.full_execution()
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("                        DISCOVERY RESULTS")
    print("=" * 70)
    if len(results) == 0:
        print("  No conjectures found within current search bounds.")
        print("  Try increasing degree or coefficient range for deeper exploration.")
    else:
        print(f"  Found {len(results)} conjecture(s)!")
        enumerator.print_results(results)
    
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n  Total runtime: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    print("=" * 70)

if __name__ == '__main__':
    main()