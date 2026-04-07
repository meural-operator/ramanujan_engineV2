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
        Analyze a discovered program by fitting its output to polynomial curves
        using numpy polyfit, then using the fitted coefficients as search bounds.
        
        This correctly handles programs like `lambda n: n**2 + 3*n + 1` by
        extracting coefficients [1, 3, 1] instead of sampling outputs at large n
        (which would give inflated values like 400+ and corrupt the bounds).
        """
        a_func = compile_lambda(prog.a_n)
        b_func = compile_lambda(prog.b_n)
        
        if a_func is None or b_func is None:
            return raw_a_bounds, raw_b_bounds
        
        # Sample the program at several n values for polyfit
        n_samples = 30
        a_vals = evaluate_sequence(a_func, n_samples)
        b_vals = evaluate_sequence(b_func, n_samples)
        
        if a_vals is None or b_vals is None:
            return raw_a_bounds, raw_b_bounds
        
        n_array = np.arange(n_samples, dtype=np.float64)
        a_deg = len(raw_a_bounds) - 1  # polynomial degree from bounds structure
        b_deg = len(raw_b_bounds) - 1
        
        try:
            # Fit polynomial of the same degree as the search domain
            a_coeffs = np.polyfit(n_array, np.array(a_vals), deg=min(a_deg, 5))
            b_coeffs = np.polyfit(n_array, np.array(b_vals), deg=min(b_deg, 5))
        except (np.linalg.LinAlgError, ValueError):
            return raw_a_bounds, raw_b_bounds
        
        # Use fitted coefficients to build tight integer bounds
        # Pad or truncate to match the expected number of coefficients
        a_coeffs_int = [int(round(c)) for c in a_coeffs]
        b_coeffs_int = [int(round(c)) for c in b_coeffs]
        
        # Pad with zeros if polyfit returned fewer coefficients
        while len(a_coeffs_int) < len(raw_a_bounds):
            a_coeffs_int.append(0)
        while len(b_coeffs_int) < len(raw_b_bounds):
            b_coeffs_int.append(0)
        
        # Build refined bounds: ±margin around each extracted coefficient
        margin = 3  # Search ±3 around each fitted coefficient
        
        refined_a = []
        for i, (lo, hi) in enumerate(raw_a_bounds):
            if i < len(a_coeffs_int):
                center = a_coeffs_int[i]
                refined_a.append([max(lo, center - margin), min(hi, center + margin)])
            else:
                refined_a.append([lo, hi])
        
        refined_b = []
        for i, (lo, hi) in enumerate(raw_b_bounds):
            if i < len(b_coeffs_int):
                center = b_coeffs_int[i]
                refined_b.append([max(lo, center - margin), min(hi, center + margin)])
            else:
                refined_b.append([lo, hi])
        
        return refined_a, refined_b
