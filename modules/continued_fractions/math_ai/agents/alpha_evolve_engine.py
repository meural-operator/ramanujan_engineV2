"""
AlphaEvolve Engine — LLM-Guided Evolutionary Discovery of Generalized Continued Fractions.

Implements a DeepMind AlphaEvolve-style evolutionary search where:
- Each individual is a program (pair of Python lambda expressions for a_n and b_n)
- Fitness = number of digits matched to a target mathematical constant  
- Mutations are proposed by a local LLM (Qwen3-Coder-30B via LM Studio)
- Crossovers intelligently combine high-fitness parents via LLM guidance
- Novel programs are periodically injected to maintain diversity

Architecture:
  Population → Selection → LLM Mutation/Crossover → Sandbox Evaluation → Archive
"""
import time
import re
import random
import json
import sqlite3
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

from modules.continued_fractions.math_ai.llm.llm_client import (
    LMStudioClient, random_mutation, _CIRCUIT_BREAKER_THRESHOLD
)
from modules.continued_fractions.math_ai.agents.program_sandbox import evaluate_gcf_fitness


@dataclass
class GCFProgram:
    """A single individual in the evolutionary population."""
    a_n: str                      # Lambda string for a(n)
    b_n: str                      # Lambda string for b(n)
    fitness: float = 0.0          # Digits of accuracy
    convergence_rate: float = 0.0 # Speed of convergence
    generation: int = 0           # When it was created
    parent_info: str = ""         # Mutation/crossover lineage
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'GCFProgram':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Seed population of known mathematical GCF structures
SEED_PROGRAMS = [
    # Pi-related
    {"a_n": "lambda n: 2*n + 1", "b_n": "lambda n: -(n**2) if n > 0 else 0",
     "parent_info": "seed:arctan_cf_pi"},
    {"a_n": "lambda n: 6 if n == 0 else 2*n - 1", "b_n": "lambda n: (2*n - 1)**2 if n > 0 else 0",
     "parent_info": "seed:pi_wallis_variant"},
    {"a_n": "lambda n: 1 if n == 0 else 2", "b_n": "lambda n: (2*n - 1)**2 if n > 0 else 1",
     "parent_info": "seed:pi_brouncker"},
     
    # E-related
    {"a_n": "lambda n: n if n > 0 else 2", "b_n": "lambda n: n if n > 0 else 1",
     "parent_info": "seed:euler_simple_cf"},
    {"a_n": "lambda n: 1 if n % 3 != 2 else 2*(n//3 + 1)", "b_n": "lambda n: 1",
     "parent_info": "seed:e_regular_cf"},
    
    # Generic polynomial structures
    {"a_n": "lambda n: n**2 + 1", "b_n": "lambda n: n + 1",
     "parent_info": "seed:quadratic_linear"},
    {"a_n": "lambda n: 2*n**2 + n + 1", "b_n": "lambda n: -(n**2 + n)",
     "parent_info": "seed:quadratic_pair"},
    {"a_n": "lambda n: n*(n+1) + 1", "b_n": "lambda n: -n*(n+2)",
     "parent_info": "seed:product_form"},
     
    # Alternating structures
    {"a_n": "lambda n: n + 1", "b_n": "lambda n: (-1)**n * (n + 1)",
     "parent_info": "seed:alternating_linear"},
    {"a_n": "lambda n: 3*n + 1", "b_n": "lambda n: -(n+1)**2",
     "parent_info": "seed:cubic_offset"},
]


class AlphaEvolveEngine:
    """
    Core evolutionary engine for GCF discovery.
    
    Maintains a population of GCF programs, evolves them through
    LLM-guided mutations and crossovers, and archives high-fitness discoveries.
    """
    def __init__(self, 
                 target_name: str,
                 target_value: float,
                 population_size: int = 30,
                 elite_fraction: float = 0.2,
                 mutation_rate: float = 0.6,
                 crossover_rate: float = 0.2,
                 novel_rate: float = 0.2,
                 n_eval_terms: int = 200,
                 archive_threshold: float = 5.0,
                 llm_client: Optional[LMStudioClient] = None,
                 db_path: Optional[str] = None,
                 disable_llm: bool = False):
        """
        Args:
            target_name: Human-readable name (e.g., "pi", "catalan")
            target_value: Numerical value of the target constant
            population_size: Number of individuals per generation
            elite_fraction: Fraction of top performers preserved unchanged
            mutation_rate: Fraction of new individuals from mutation
            crossover_rate: Fraction of new individuals from crossover
            novel_rate: Fraction of new individuals from LLM novel proposals
            n_eval_terms: Number of GCF terms for fitness evaluation
            archive_threshold: Minimum digits accuracy to archive a discovery
            llm_client: LMStudioClient instance (auto-created if None)
            db_path: SQLite path for persistent evolution archive
        """
        self.target_name = target_name
        self.target_value = target_value
        self.pop_size = population_size
        self.elite_frac = elite_fraction
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.novel_rate = novel_rate
        self.n_eval_terms = n_eval_terms
        self.archive_threshold = archive_threshold
        
        self.llm = llm_client or LMStudioClient()
        self.disable_llm = disable_llm
        # NOTE: Do NOT cache self.llm.is_available() here.
        # The client has its own TTL cache. We call it live each generation
        # so the engine picks up LM Studio restarts mid-campaign.
        
        self.population: List[GCFProgram] = []
        self.archive: List[GCFProgram] = []
        self.generation = 0
        self.best_ever: Optional[GCFProgram] = None
        
        # Statistics
        self.stats_history: List[dict] = []
        
        # SQLite persistence
        self.db_path = db_path or f"evolve_{target_name}.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite archive database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS discoveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_name TEXT,
            a_n TEXT,
            b_n TEXT,
            fitness REAL,
            convergence_rate REAL,
            generation INTEGER,
            parent_info TEXT,
            timestamp REAL
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS evolution_log (
            generation INTEGER,
            best_fitness REAL,
            avg_fitness REAL,
            population_size INTEGER,
            llm_mutations INTEGER,
            random_mutations INTEGER,
            timestamp REAL
        )''')
        conn.commit()
        conn.close()
    
    def initialize_population(self):
        """Bootstrap the population with seed programs + random variants."""
        print(f"\n[AlphaEvolve] Initializing population for [{self.target_name}]...")
        
        # Add seed programs
        for seed in SEED_PROGRAMS:
            prog = GCFProgram(
                a_n=seed['a_n'], 
                b_n=seed['b_n'],
                parent_info=seed['parent_info']
            )
            self.population.append(prog)
        
        # Fill remaining slots with random polynomial mutations of seeds
        while len(self.population) < self.pop_size:
            base = random.choice(SEED_PROGRAMS)
            mutated_a, mutated_b = random_mutation(base['a_n'], base['b_n'])
            prog = GCFProgram(
                a_n=mutated_a,
                b_n=mutated_b,
                parent_info="seed:random_init"
            )
            self.population.append(prog)
        
        # Evaluate fitness of initial population
        self._evaluate_population()
        
        print(f"[AlphaEvolve] Population initialized: {len(self.population)} individuals")
        if self.best_ever:
            print(f"[AlphaEvolve] Best initial: {self.best_ever.fitness:.2f} digits")
            print(f"              a(n) = {self.best_ever.a_n}")
            print(f"              b(n) = {self.best_ever.b_n}")
    
    def _evaluate_population(self):
        """Evaluate fitness for all individuals without a fitness score."""
        for prog in self.population:
            if prog.fitness == 0.0:
                result = evaluate_gcf_fitness(
                    prog.a_n, prog.b_n, 
                    self.target_value, 
                    n_terms=self.n_eval_terms
                )
                if result['valid']:
                    prog.fitness = result['fitness']
                    prog.convergence_rate = result['convergence_rate']
                else:
                    prog.fitness = -1.0  # Mark as invalid
        
        # Remove invalid programs
        self.population = [p for p in self.population if p.fitness >= 0]
        
        # Sort by fitness (descending)
        self.population.sort(key=lambda p: p.fitness, reverse=True)
        
        # Update best ever
        if self.population and (self.best_ever is None or self.population[0].fitness > self.best_ever.fitness):
            self.best_ever = GCFProgram(**asdict(self.population[0]))
    
    def _tournament_select(self, k: int = 3) -> GCFProgram:
        """Tournament selection: pick k random individuals, return the best."""
        candidates = random.sample(self.population, min(k, len(self.population)))
        return max(candidates, key=lambda p: p.fitness)
    
    @staticmethod
    def _interpolate_lambda(code_a: str, code_b: str) -> str:
        """
        Interpolate numeric coefficients between two lambda expression strings.
        Uses the first (code_a) as the structural template and blends coefficients
        from the second (code_b) via uniform random interpolation weights.
        """
        nums_a = re.findall(r'(?<!\w)(\d+)', code_a)
        nums_b = re.findall(r'(?<!\w)(\d+)', code_b)
        
        if not nums_a:
            return code_a  # No coefficients to interpolate
        
        result = code_a
        for i, num_str in enumerate(nums_a):
            val_a = int(num_str)
            # Pick a corresponding coefficient from parent B (cycling if shorter)
            val_b = int(nums_b[i % len(nums_b)]) if nums_b else val_a
            # Random interpolation weight per coefficient
            alpha = random.random()
            blended = int(round(alpha * val_a + (1 - alpha) * val_b))
            blended = max(0, blended)  # Keep non-negative for safety
            # Replace only the first remaining occurrence
            result = result.replace(num_str, str(blended), 1)
        
        return result
    
    def _interpolate_crossover(self, parent_a: GCFProgram, parent_b: GCFProgram) -> tuple:
        """
        Coefficient-level interpolation crossover.
        Uses the fitter parent as the structural template and blends numeric
        coefficients from both parents. This preserves mathematical coherence
        between a(n) and b(n) instead of naively swapping components.
        """
        # Use the fitter parent as the structural template
        if parent_a.fitness >= parent_b.fitness:
            template, donor = parent_a, parent_b
        else:
            template, donor = parent_b, parent_a
        
        try:
            child_a_n = self._interpolate_lambda(template.a_n, donor.a_n)
            child_b_n = self._interpolate_lambda(template.b_n, donor.b_n)
            return (child_a_n, child_b_n)
        except Exception:
            # If interpolation fails, return the fitter parent with a small mutation
            return random_mutation(template.a_n, template.b_n)
    
    def _archive_discoveries(self):
        """Archive any programs exceeding the discovery threshold."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        for prog in self.population:
            if prog.fitness >= self.archive_threshold:
                # Check if this exact program is already archived
                c.execute("SELECT id FROM discoveries WHERE a_n = ? AND b_n = ?",
                          (prog.a_n, prog.b_n))
                if not c.fetchone():
                    c.execute(
                        "INSERT INTO discoveries VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (self.target_name, prog.a_n, prog.b_n, prog.fitness,
                         prog.convergence_rate, prog.generation, prog.parent_info,
                         time.time())
                    )
                    self.archive.append(prog)
                    print(f"\n[!!] ARCHIVED DISCOVERY: {prog.fitness:.2f} digits for {self.target_name}")
                    print(f"     a(n) = {prog.a_n}")
                    print(f"     b(n) = {prog.b_n}")
        
        conn.commit()
        conn.close()
    
    def _log_generation(self, llm_mutations: int, random_mutations: int):
        """Log generation statistics."""
        if not self.population:
            return
        
        fitnesses = [p.fitness for p in self.population]
        stats = {
            'generation': self.generation,
            'best_fitness': max(fitnesses),
            'avg_fitness': sum(fitnesses) / len(fitnesses),
            'population_size': len(self.population),
            'llm_mutations': llm_mutations,
            'random_mutations': random_mutations,
        }
        self.stats_history.append(stats)
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO evolution_log VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (stats['generation'], stats['best_fitness'], stats['avg_fitness'],
                   stats['population_size'], llm_mutations, random_mutations, time.time()))
        conn.commit()
        conn.close()
    
    def evolve_generation(self) -> dict:
        """
        Run a single generation of evolution.
        
        Performance optimizations vs. the naive sequential loop:
          - LLM mutations submitted in parallel via ThreadPoolExecutor
            (GIL releases during urllib I/O → N calls ≈ 1 round-trip)
          - Circuit breaker: 3 consecutive failures → skip remaining LLM calls
          - LRU prompt cache: converged populations avoid redundant LLM calls
          - n_novel off-by-one fixed (uses max(0, ...) not max(1, ...))
          - Archive summary hoisted outside the novel loop
        
        Returns:
            Dict with generation statistics.
        """
        self.generation += 1
        
        # Reset circuit breaker for this generation
        self.llm.reset_circuit()
        
        # Live availability check (uses client TTL cache, NOT a stale __init__ value)
        # In ablation mode (disable_llm=True), always report LLM as down
        llm_up = (not self.disable_llm) and self.llm.is_available()
        
        # 1. Elitism — preserve top performers unchanged
        n_elite = max(1, int(self.pop_size * self.elite_frac))
        next_gen = [GCFProgram(**asdict(p)) for p in self.population[:n_elite]]
        for p in next_gen:
            p.generation = self.generation
        
        n_mutations = int(self.pop_size * self.mutation_rate)
        n_crossovers = int(self.pop_size * self.crossover_rate)
        n_novel = max(0, self.pop_size - n_elite - n_mutations - n_crossovers)
        
        llm_mutations = 0
        random_mutations_count = 0
        
        # ── 2. MUTATIONS — parallel LLM calls ──
        # Select parents first, then batch all LLM calls in parallel
        mutation_parents = [self._tournament_select() for _ in range(n_mutations)]
        
        # Decide which parents get LLM vs random (80% LLM, 20% random)
        llm_parent_indices = []
        random_parent_indices = []
        for i, parent in enumerate(mutation_parents):
            if llm_up and self.llm.circuit_ok and random.random() < 0.8:
                llm_parent_indices.append(i)
            else:
                random_parent_indices.append(i)
        
        # Submit all LLM mutation requests in parallel
        llm_results = [None] * n_mutations
        if llm_parent_indices:
            llm_parents_batch = [
                {'a_n': mutation_parents[i].a_n, 'b_n': mutation_parents[i].b_n,
                 'fitness': mutation_parents[i].fitness}
                for i in llm_parent_indices
            ]
            batch_results = self.llm.propose_mutations_parallel(
                llm_parents_batch, self.target_name
            )
            for j, idx in enumerate(llm_parent_indices):
                llm_results[idx] = batch_results[j]
        
        # Assemble mutation children
        for i, parent in enumerate(mutation_parents):
            mutated = llm_results[i]
            if mutated:
                llm_mutations += 1
            else:
                mutated = random_mutation(parent.a_n, parent.b_n)
                random_mutations_count += 1
            
            child = GCFProgram(
                a_n=mutated[0], b_n=mutated[1],
                generation=self.generation,
                parent_info=f"mutation_of:{parent.a_n[:30]}..."
            )
            next_gen.append(child)
        
        # ── 3. CROSSOVERS — parallel LLM calls ──
        crossover_pairs = [
            (self._tournament_select(), self._tournament_select())
            for _ in range(n_crossovers)
        ]
        
        crossed_results = [None] * n_crossovers
        if llm_up and self.llm.circuit_ok and crossover_pairs:
            pair_dicts = [
                (pa.to_dict(), pb.to_dict())
                for pa, pb in crossover_pairs
            ]
            crossed_results = self.llm.propose_crossovers_parallel(
                pair_dicts, self.target_name
            )
        
        for i, (parent_a, parent_b) in enumerate(crossover_pairs):
            crossed = crossed_results[i]
            if crossed:
                llm_mutations += 1
            else:
                crossed = self._interpolate_crossover(parent_a, parent_b)
                random_mutations_count += 1
            
            child = GCFProgram(
                a_n=crossed[0], b_n=crossed[1],
                generation=self.generation,
                parent_info="crossover"
            )
            next_gen.append(child)
        
        # ── 4. NOVEL PROPOSALS ──
        # Hoist archive summary outside the loop (critique #6)
        archive_summary = (
            ", ".join(f"a(n)={p.a_n}" for p in self.archive[-5:])
            if self.archive else ""
        )
        
        # Fix off-by-one: n_novel=0 means no novel calls (critique #4)
        for _ in range(n_novel):
            novel = None
            if llm_up and self.llm.circuit_ok:
                novel = self.llm.propose_novel(
                    self.target_name, self.target_value, archive_summary
                )
                if novel:
                    llm_mutations += 1
            
            if novel is None:
                base = random.choice(SEED_PROGRAMS)
                novel = random_mutation(base['a_n'], base['b_n'])
                novel = random_mutation(novel[0], novel[1])
                random_mutations_count += 1
            
            child = GCFProgram(
                a_n=novel[0], b_n=novel[1],
                generation=self.generation,
                parent_info="novel"
            )
            next_gen.append(child)
        
        # 5. Evaluate and sort
        self.population = next_gen
        self._evaluate_population()
        
        # Trim to population size
        self.population = self.population[:self.pop_size]
        
        # 6. Archive discoveries
        self._archive_discoveries()
        
        # 7. Log
        self._log_generation(llm_mutations, random_mutations_count)
        
        return self.stats_history[-1] if self.stats_history else {}
    
    def run(self, max_generations: int = 100, verbose: bool = True):
        """
        Run the full evolutionary search.
        
        Args:
            max_generations: Maximum number of generations to evolve
            verbose: Print progress each generation
        """
        print("=" * 70)
        print(f"   AlphaEvolve: LLM-Guided Evolutionary GCF Discovery")
        print(f"   Target: {self.target_name} ≈ {self.target_value:.15f}")
        _llm_status = self.llm.is_available()
        _llm_mode = 'DISABLED (ablation)' if self.disable_llm else (
            'Qwen3-Coder-30B via LM Studio' if _llm_status else 'UNAVAILABLE (random fallback)')
        print(f"   LLM: {_llm_mode}")
        print(f"   Population: {self.pop_size} | Generations: {max_generations}")
        print(f"   LLM Timeout: {self.llm.timeout}s | Circuit Breaker: {_CIRCUIT_BREAKER_THRESHOLD} failures")
        print("=" * 70)
        
        self.initialize_population()
        
        start_time = time.time()
        
        for gen in range(max_generations):
            gen_start = time.time()
            stats = self.evolve_generation()
            gen_time = time.time() - gen_start
            
            if verbose and stats:
                best = self.population[0] if self.population else None
                print(
                    f"  Gen {stats['generation']:4d} | "
                    f"Best: {stats['best_fitness']:6.2f} | "
                    f"Avg: {stats['avg_fitness']:5.2f} | "
                    f"LLM: {stats['llm_mutations']} | "
                    f"Pop: {stats['population_size']} | "
                    f"{gen_time:.1f}s"
                    + (f" | a(n)={best.a_n[:50]}" if best else "")
                )
            
            # Early termination if we found a very high quality match
            if self.best_ever and self.best_ever.fitness >= 12.0:
                print(f"\n[AlphaEvolve] BREAKTHROUGH: {self.best_ever.fitness:.2f} digits matched!")
                break
        
        elapsed = time.time() - start_time
        h, r = divmod(elapsed, 3600)
        m, s = divmod(r, 60)
        
        # Print cache stats if LLM was used
        if not self.disable_llm:
            cs = self.llm.cache_stats
            print(f"\n   LLM Cache: {cs['hits']} hits / {cs['misses']} misses ({cs['hit_rate']} hit rate)")
        
        print(f"\n{'=' * 70}")
        print(f"   EVOLUTION COMPLETE")
        print(f"{'=' * 70}")
        print(f"   Generations: {self.generation}")
        print(f"   Total Time: {int(h)}h {int(m)}m {s:.1f}s")
        print(f"   Best Fitness: {self.best_ever.fitness:.2f} digits" if self.best_ever else "   No valid programs found")
        print(f"   Archive Size: {len(self.archive)} discoveries")
        if self.best_ever:
            print(f"   Best Program:")
            print(f"     a(n) = {self.best_ever.a_n}")
            print(f"     b(n) = {self.best_ever.b_n}")
        print(f"{'=' * 70}")
        
        return self.archive

    @staticmethod
    def run_ablation_study(target_name: str, target_value: float, 
                           generations: int = 30, population_size: int = 30,
                           seed: int = 42):
        """
        Run a controlled ablation study: LLM-guided vs random-only evolution.
        
        Uses identical random seeds so both runs start from the same initial 
        population and face the same tournament selection sequence. This isolates
        the LLM's contribution from other random factors.
        
        Prints a comparative table with:
          - Best fitness achieved
          - Generation at which peak fitness was reached
          - Wall-clock time (LLM has I/O overhead; random is pure CPU)
          - Cost efficiency: seconds per digit of improvement
        
        Usage:
            AlphaEvolveEngine.run_ablation_study("euler_mascheroni", 0.5772156649)
        """
        print("=" * 70)
        print("   ABLATION STUDY: LLM-Guided vs Random-Only Evolution")
        print(f"   Target: {target_name} ≈ {target_value:.15f}")
        print(f"   Generations: {generations} | Population: {population_size} | Seed: {seed}")
        print("=" * 70)
        
        results = {}
        
        for mode, disable in [("LLM-Guided", False), ("Random-Only", True)]:
            print(f"\n{'─' * 40}")
            print(f"   Running: {mode}")
            print(f"{'─' * 40}")
            
            random.seed(seed)
            
            engine = AlphaEvolveEngine(
                target_name=target_name,
                target_value=target_value,
                population_size=population_size,
                db_path=f"ablation_{mode.lower().replace('-','_')}_{target_name}.db",
                disable_llm=disable,
            )
            
            engine.initialize_population()
            start = time.time()
            
            best_per_gen = []
            for g in range(generations):
                stats = engine.evolve_generation()
                best_fitness = stats.get('best_fitness', 0.0) if stats else 0.0
                best_per_gen.append(best_fitness)
                
                if best_fitness >= 12.0:
                    break
            
            elapsed = time.time() - start
            
            peak_fitness = max(best_per_gen) if best_per_gen else 0.0
            peak_gen = best_per_gen.index(peak_fitness) + 1 if best_per_gen else 0
            
            results[mode] = {
                'peak_fitness': peak_fitness,
                'peak_gen': peak_gen,
                'elapsed': elapsed,
                'final_gen': len(best_per_gen),
                'archive_size': len(engine.archive),
                'curve': best_per_gen,
            }
        
        # Print comparison table
        print(f"\n{'=' * 70}")
        print(f"   ABLATION RESULTS")
        print(f"{'=' * 70}")
        print(f"   {'Metric':<30} {'LLM-Guided':>15} {'Random-Only':>15}")
        print(f"   {'─' * 60}")
        
        llm = results["LLM-Guided"]
        rnd = results["Random-Only"]
        
        print(f"   {'Peak Fitness (digits)':<30} {llm['peak_fitness']:>15.2f} {rnd['peak_fitness']:>15.2f}")
        print(f"   {'Peak at Generation':<30} {llm['peak_gen']:>15d} {rnd['peak_gen']:>15d}")
        print(f"   {'Wall-Clock Time (s)':<30} {llm['elapsed']:>15.1f} {rnd['elapsed']:>15.1f}")
        print(f"   {'Discoveries Archived':<30} {llm['archive_size']:>15d} {rnd['archive_size']:>15d}")
        
        # Cost per digit
        llm_cpd = llm['elapsed'] / max(0.01, llm['peak_fitness'])
        rnd_cpd = rnd['elapsed'] / max(0.01, rnd['peak_fitness'])
        print(f"   {'Seconds per Digit':<30} {llm_cpd:>15.1f} {rnd_cpd:>15.1f}")
        
        # Verdict
        delta = llm['peak_fitness'] - rnd['peak_fitness']
        speedup = rnd['elapsed'] / max(0.01, llm['elapsed'])
        
        print(f"\n   {'─' * 60}")
        if delta > 0.5:
            print(f"   VERDICT: LLM adds +{delta:.2f} digits — significant contribution.")
            print(f"   Recommendation: KEEP LLM for the compute campaign.")
        elif delta > 0.1:
            print(f"   VERDICT: LLM adds +{delta:.2f} digits — marginal benefit.")
            print(f"   Recommendation: Keep LLM but increase random mutation diversity.")
        else:
            print(f"   VERDICT: LLM adds {delta:+.2f} digits — NO measurable benefit.")
            print(f"   Random-only is {speedup:.1f}x faster. LLM is pure overhead.")
            print(f"   Recommendation: DISABLE LLM and reallocate budget to GPU enumeration.")
        print(f"{'=' * 70}")
        
        return results

