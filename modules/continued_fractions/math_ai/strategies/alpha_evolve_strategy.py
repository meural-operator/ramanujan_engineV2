"""
AlphaEvolve Strategy — BoundingStrategy plugin for the UniversalPipelineRouter.

Runs a short evolutionary burst via the AlphaEvolve engine before GPU exhaustion,
converting the best-discovered program structures into polynomial coefficient bounds
for the CartesianProductPolyDomain.
"""
import numpy as np
from typing import Tuple, List

from core.interfaces.base_strategy import BoundingStrategy
from modules.continued_fractions.math_ai.agents.alpha_evolve_engine import AlphaEvolveEngine
from modules.continued_fractions.math_ai.llm.llm_client import LMStudioClient
from modules.continued_fractions.math_ai.agents.program_sandbox import compile_lambda, evaluate_sequence


class AlphaEvolveStrategy(BoundingStrategy):
    """
    Plugin wrapper for the LLM-guided evolutionary search.
    Runs a configurable number of evolution generations, then extracts 
    the best-discovered polynomial patterns to refine GPU search bounds.
    """
    def __init__(self, target_name: str = "unknown", target_value: float = 0.0,
                 generations: int = 20, population_size: int = 20):
        self.target_name = target_name
        self.target_value = target_value
        self.generations = generations
        self.population_size = population_size
        self.llm = LMStudioClient()
        self._llm_checked = False
    
    @property
    def strategy_name(self) -> str:
        return "alpha_evolve_llm"
    
    def prune_bounds(self, raw_a_bounds: List[List[int]], raw_b_bounds: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Runs evolutionary search and uses discovered programs to inform polynomial bounds.
        Falls back to identity (no pruning) if evolution yields nothing useful.
        """
        if not self._llm_checked:
            self._llm_checked = True
            if not self.llm.is_available():
                print(f"[AlphaEvolve] LM Studio not reachable. Falling back to original bounds.")
                return raw_a_bounds, raw_b_bounds
        
        print(f"[AlphaEvolve] Running {self.generations}-generation evolutionary burst...")
        
        engine = AlphaEvolveEngine(
            target_name=self.target_name,
            target_value=self.target_value,
            population_size=self.population_size,
            max_generations=self.generations,
            llm_client=self.llm
        )
        
        engine.initialize_population()
        
        for _ in range(self.generations):
            engine.evolve_generation()
        
        # Extract coefficient patterns from top programs
        if engine.best_ever and engine.best_ever.fitness > 2.0:
            print(f"[AlphaEvolve] Best evolved: {engine.best_ever.fitness:.2f} digits")
            print(f"  a(n) = {engine.best_ever.a_n}")
            print(f"  b(n) = {engine.best_ever.b_n}")
            
            # Analyze the discovered program to extract approximate polynomial coefficients
            refined_a, refined_b = self._extract_bounds_from_program(
                engine.best_ever, raw_a_bounds, raw_b_bounds
            )
            return refined_a, refined_b
        
        print(f"[AlphaEvolve] No high-fitness programs found. Using original bounds.")
        return raw_a_bounds, raw_b_bounds
    
    def _extract_bounds_from_program(self, prog, raw_a_bounds, raw_b_bounds):
        """
        Analyze a discovered program's numerical outputs and estimate 
        polynomial coefficient bounds that would reproduce similar sequences.
        """
        a_func = compile_lambda(prog.a_n)
        b_func = compile_lambda(prog.b_n)
        
        if a_func is None or b_func is None:
            return raw_a_bounds, raw_b_bounds
        
        # Sample the program at several n values
        n_samples = 20
        a_vals = evaluate_sequence(a_func, n_samples)
        b_vals = evaluate_sequence(b_func, n_samples)
        
        if a_vals is None or b_vals is None:
            return raw_a_bounds, raw_b_bounds
        
        # Use the range of generated values to inform polynomial bounds
        # This is a rough heuristic — the evolved program tells us what
        # coefficient magnitudes are mathematically productive
        a_max = max(abs(v) for v in a_vals if abs(v) < 1e6)
        b_max = max(abs(v) for v in b_vals if abs(v) < 1e6)
        
        a_radius = max(3, int(a_max * 1.5))
        b_radius = max(3, int(b_max * 1.5))
        
        refined_a = []
        for lo, hi in raw_a_bounds:
            mid = (lo + hi) // 2
            refined_a.append([max(lo, mid - a_radius), min(hi, mid + a_radius)])
        
        refined_b = []
        for lo, hi in raw_b_bounds:
            mid = (lo + hi) // 2
            refined_b.append([max(lo, mid - b_radius), min(hi, mid + b_radius)])
        
        return refined_a, refined_b
