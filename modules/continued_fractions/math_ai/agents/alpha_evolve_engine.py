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
import random
import json
import sqlite3
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

from modules.continued_fractions.math_ai.llm.llm_client import LMStudioClient, random_mutation
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
                 db_path: Optional[str] = None):
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
        self.llm_available = self.llm.is_available()
        
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
        
        Returns:
            Dict with generation statistics.
        """
        self.generation += 1
        
        # 1. Elitism — preserve top performers unchanged
        n_elite = max(1, int(self.pop_size * self.elite_frac))
        next_gen = [GCFProgram(**asdict(p)) for p in self.population[:n_elite]]
        for p in next_gen:
            p.generation = self.generation
        
        n_mutations = int(self.pop_size * self.mutation_rate)
        n_crossovers = int(self.pop_size * self.crossover_rate)
        n_novel = self.pop_size - n_elite - n_mutations - n_crossovers
        
        llm_mutations = 0
        random_mutations_count = 0
        
        # 2. Mutations — select parent, mutate via LLM or random
        for _ in range(n_mutations):
            parent = self._tournament_select()
            
            mutated = None
            if self.llm_available and random.random() < 0.8:  # 80% LLM, 20% random
                mutated = self.llm.propose_mutation(
                    parent.a_n, parent.b_n,
                    self.target_name, parent.fitness
                )
                if mutated:
                    llm_mutations += 1
            
            if mutated is None:
                mutated = random_mutation(parent.a_n, parent.b_n)
                random_mutations_count += 1
            
            child = GCFProgram(
                a_n=mutated[0], b_n=mutated[1],
                generation=self.generation,
                parent_info=f"mutation_of:{parent.a_n[:30]}..."
            )
            next_gen.append(child)
        
        # 3. Crossovers — select two parents, combine via LLM
        for _ in range(n_crossovers):
            parent_a = self._tournament_select()
            parent_b = self._tournament_select()
            
            crossed = None
            if self.llm_available:
                crossed = self.llm.propose_crossover(
                    parent_a.to_dict(), parent_b.to_dict(),
                    self.target_name
                )
                if crossed:
                    llm_mutations += 1
            
            if crossed is None:
                # Simple crossover: take a_n from one parent, b_n from other
                if random.random() < 0.5:
                    crossed = (parent_a.a_n, parent_b.b_n)
                else:
                    crossed = (parent_b.a_n, parent_a.b_n)
                random_mutations_count += 1
            
            child = GCFProgram(
                a_n=crossed[0], b_n=crossed[1],
                generation=self.generation,
                parent_info="crossover"
            )
            next_gen.append(child)
        
        # 4. Novel proposals — ask LLM for entirely new programs
        for _ in range(max(1, n_novel)):
            novel = None
            if self.llm_available:
                archive_summary = ", ".join(
                    f"a(n)={p.a_n}" for p in self.archive[-5:]
                ) if self.archive else ""
                novel = self.llm.propose_novel(
                    self.target_name, self.target_value, archive_summary
                )
                if novel:
                    llm_mutations += 1
            
            if novel is None:
                # Random novel program
                base = random.choice(SEED_PROGRAMS)
                novel = random_mutation(base['a_n'], base['b_n'])
                # Double mutate for more novelty
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
        print(f"   LLM: {'Qwen3-Coder-30B via LM Studio' if self.llm_available else 'UNAVAILABLE (random fallback)'}")
        print(f"   Population: {self.pop_size} | Generations: {max_generations}")
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
