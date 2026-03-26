"""
Ramanujan@Home: AlphaEvolve Discovery Miner

Standalone long-running evolutionary search for novel Generalized Continued Fractions.
Uses a local LLM (Qwen3-Coder-30B via LM Studio) to evolve program structures
that converge to mathematical constants.

This runs independently of the GPU brute-force pipeline — fitness evaluation
uses CPU-based GCF recurrence, while the GPU serves the LLM.

Usage:
    python scripts/evolve_miner.py [--target pi|e|catalan|gamma] [--generations 200]
"""
import sys
import os
import argparse
import mpmath

# Ensure the framework root resolves safely
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from modules.continued_fractions.math_ai.agents.alpha_evolve_engine import AlphaEvolveEngine
from modules.continued_fractions.math_ai.llm.llm_client import LMStudioClient

# Target constants registry
TARGETS = {
    "pi": ("π (Pi)", float(mpmath.pi)),
    "e": ("e (Euler's number)", float(mpmath.e)),
    "catalan": ("G (Catalan's constant)", float(mpmath.catalan)),
    "gamma": ("γ (Euler-Mascheroni)", float(mpmath.euler)),
    "golden_ratio": ("φ (Golden Ratio)", float(mpmath.phi)),
    "ln2": ("ln(2)", float(mpmath.log(2))),
    "sqrt2": ("√2", float(mpmath.sqrt(2))),
    "zeta3": ("ζ(3) (Apéry's constant)", float(mpmath.zeta(3))),
}


def main():
    parser = argparse.ArgumentParser(description="AlphaEvolve: LLM-Guided GCF Discovery")
    parser.add_argument("--target", type=str, default="pi",
                        choices=list(TARGETS.keys()),
                        help="Target mathematical constant")
    parser.add_argument("--generations", type=int, default=100,
                        help="Number of evolution generations")
    parser.add_argument("--population", type=int, default=30,
                        help="Population size per generation")
    parser.add_argument("--eval-terms", type=int, default=200,
                        help="Number of GCF terms for fitness evaluation")
    parser.add_argument("--archive-threshold", type=float, default=5.0,
                        help="Minimum digits of accuracy to archive a discovery")
    parser.add_argument("--lm-studio-url", type=str, default="http://127.0.0.1:1234",
                        help="LM Studio API URL")
    parser.add_argument("--model", type=str, default="qwen/qwen3-coder-30b",
                        help="LM Studio model identifier")
    args = parser.parse_args()
    
    target_name, target_value = TARGETS[args.target]
    
    print("=" * 70)
    print("   Ramanujan@Home: AlphaEvolve Discovery Miner")
    print("=" * 70)
    print(f"   Target: {target_name} ≈ {target_value:.15f}")
    print(f"   Generations: {args.generations}")
    print(f"   Population: {args.population}")
    print(f"   LM Studio: {args.lm_studio_url}")
    print(f"   Model: {args.model}")
    print()
    
    # Initialize LLM client
    llm = LMStudioClient(base_url=args.lm_studio_url, model=args.model)
    
    if llm.is_available():
        print("[+] LM Studio connection established. LLM-guided evolution active.")
    else:
        print("[!] LM Studio not reachable. Falling back to random mutations.")
        print("    Start LM Studio and load a model to enable intelligent evolution.")
    print()
    
    # Initialize and run engine
    engine = AlphaEvolveEngine(
        target_name=args.target,
        target_value=target_value,
        population_size=args.population,
        n_eval_terms=args.eval_terms,
        archive_threshold=args.archive_threshold,
        llm_client=llm,
        db_path=os.path.join(repo_root, f"evolve_{args.target}.db")
    )
    
    discoveries = engine.run(max_generations=args.generations, verbose=True)
    
    # Print final discoveries
    if discoveries:
        print(f"\n{'=' * 70}")
        print(f"   DISCOVERIES FOR {target_name}")
        print(f"{'=' * 70}")
        for i, d in enumerate(discoveries, 1):
            print(f"\n   #{i}: {d.fitness:.2f} digits of {args.target}")
            print(f"     a(n) = {d.a_n}")
            print(f"     b(n) = {d.b_n}")
            print(f"     Source: {d.parent_info}")
    else:
        print("\n   No discoveries archived in this run.")
        print("   Try increasing --generations or --population for longer searches.")
    
    return 0 if discoveries else 1


if __name__ == '__main__':
    sys.exit(main())
