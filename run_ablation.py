"""
Ablation Study: LLM-Guided vs Random-Only Evolution
Tests whether Qwen3-Coder-30B provides measurable benefit over random mutation.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.continued_fractions.math_ai.agents.alpha_evolve_engine import AlphaEvolveEngine

if __name__ == '__main__':
    # Run the ablation study targeting Euler-Mascheroni constant
    # 15 generations is enough to see convergence trends without burning hours
    results = AlphaEvolveEngine.run_ablation_study(
        target_name="euler_mascheroni",
        target_value=0.5772156649015328,
        generations=15,
        population_size=20,
        seed=42
    )
