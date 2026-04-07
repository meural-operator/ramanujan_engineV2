"""
LM Studio Client for AlphaEvolve — LLM-Guided Evolutionary GCF Discovery.

Connects to a local LM Studio instance running Qwen3-Coder-30B to propose
mathematically-motivated mutations and crossovers for GCF program evolution.

Performance architecture:
  - Parallel LLM calls via concurrent.futures.ThreadPoolExecutor
    (GIL releases during I/O-bound urllib requests)
  - Circuit breaker: 3 consecutive failures → skip remaining LLM calls for this generation
  - LRU prompt cache: hash(prompt) → response, eliminates redundant calls
    when converged populations re-generate identical parents
  - TTL on availability cache: negative results expire after 5 minutes
"""
import json
import re
import random
import time
import hashlib
import urllib.request
import urllib.error
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, List, Dict

# Default LM Studio configuration
DEFAULT_BASE_URL = "http://127.0.0.1:1234"
DEFAULT_MODEL = "qwen/qwen3-coder-30b"

# Cache TTL for availability checks (seconds)
_AVAILABILITY_CACHE_TTL = 300  # Re-check every 5 minutes

# Circuit breaker threshold
_CIRCUIT_BREAKER_THRESHOLD = 3  # Consecutive failures before tripping

# LRU cache size for prompt responses
_PROMPT_CACHE_SIZE = 256

# Max parallel LLM workers (bounded to avoid overwhelming LM Studio)
_MAX_LLM_WORKERS = 4

MUTATION_SYSTEM_PROMPT = """You are a mathematical research assistant specializing in continued fractions and number theory.

Your task is to propose MUTATIONS to Python lambda expressions that generate sequences a(n) and b(n) for Generalized Continued Fractions (GCFs). The GCF is defined as:

  target = a(0) + b(1)/(a(1) + b(2)/(a(2) + b(3)/(a(3) + ...)))

Your mutations should be mathematically motivated. Consider:
- Polynomial modifications (change degree, shift coefficients)
- Introducing factorials, binomial coefficients, or powers
- Modular arithmetic patterns (n % k)
- Alternating signs (-1)**n
- Products of linear terms n*(n+1), n*(2*n+1)
- Known number-theoretic sequences

CRITICAL RULES:
1. Output ONLY valid Python lambda expressions, one per line
2. Each lambda takes a single integer argument n (n >= 0)
3. Use only: +, -, *, /, //, **, %, abs(), min(), max()
4. Do NOT use imports, function definitions, or multi-line expressions
5. Do NOT use math.factorial for large n — keep expressions bounded
6. Respond with exactly 2 lines: first for a(n), second for b(n)
7. Do NOT include any explanation, think tags, or anything else — ONLY the two lambda lines"""

CROSSOVER_SYSTEM_PROMPT = """You are a mathematical research assistant specializing in continued fractions.

Given two parent GCF programs that converge to a mathematical constant, create a CHILD program by intelligently combining elements from both parents.

CRITICAL RULES:
1. Output ONLY valid Python lambda expressions, one per line  
2. Each lambda takes a single integer argument n (n >= 0)
3. Use only: +, -, *, /, //, **, %, abs(), min(), max()
4. Respond with exactly 2 lines: first for a(n), second for b(n)
5. Do NOT include any explanation, think tags, or anything else — ONLY the two lambda lines"""


class _LRUCache:
    """Simple LRU cache using OrderedDict for prompt→response deduplication."""
    
    def __init__(self, maxsize: int = _PROMPT_CACHE_SIZE):
        self._cache = OrderedDict()
        self._maxsize = maxsize
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[str]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None
    
    def put(self, key: str, value: str):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
        self._cache[key] = value
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LMStudioClient:
    """
    High-performance client for LM Studio's local inference API.
    
    Architecture:
      - ThreadPoolExecutor for parallel I/O-bound LLM calls
      - Circuit breaker: trips after 3 consecutive failures, resets on success
      - LRU prompt cache: eliminates redundant calls from converged populations
      - TTL availability cache: re-probes after 5 minutes on negative results
    """
    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 timeout: int = 30, max_workers: int = _MAX_LLM_WORKERS):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout  # Reduced from 60s to 30s — fail faster
        self.max_workers = max_workers
        self._available = None
        self._available_ts = 0.0
        
        # Circuit breaker state
        self._consecutive_failures = 0
        self._circuit_tripped = False
        
        # LRU prompt cache
        self._cache = _LRUCache(maxsize=_PROMPT_CACHE_SIZE)
        
        # Thread pool (lazy init)
        self._executor = None
    
    def _get_executor(self) -> ThreadPoolExecutor:
        """Lazy-init thread pool to avoid cost if LLM is never used."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor
    
    def is_available(self) -> bool:
        """
        Check if LM Studio is reachable with TTL cache.
        Negative results expire after 5 minutes.
        """
        now = time.time()
        if self._available is None or (not self._available and (now - self._available_ts) > _AVAILABILITY_CACHE_TTL):
            try:
                req = urllib.request.Request(f"{self.base_url}/api/v1/models", method='GET')
                urllib.request.urlopen(req, timeout=5)
                self._available = True
            except Exception:
                self._available = False
            self._available_ts = now
        return self._available
    
    @property
    def circuit_ok(self) -> bool:
        """Check if the circuit breaker allows calls."""
        return not self._circuit_tripped
    
    def reset_circuit(self):
        """Reset circuit breaker at the start of each generation."""
        self._circuit_tripped = False
        self._consecutive_failures = 0
    
    def _record_success(self):
        """Record a successful LLM call — resets the consecutive failure counter."""
        self._consecutive_failures = 0
    
    def _record_failure(self):
        """Record a failed LLM call — trips the breaker after threshold."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
            self._circuit_tripped = True
            print(f"[AlphaEvolve] Circuit breaker TRIPPED after {self._consecutive_failures} "
                  f"consecutive LLM failures — skipping remaining LLM calls this generation")
    
    @staticmethod
    def _hash_prompt(system: str, user: str) -> str:
        """Compute a cache key from the prompt content."""
        raw = f"{system}|||{user}"
        return hashlib.md5(raw.encode('utf-8')).hexdigest()
    
    def _chat(self, system_prompt: str, user_input: str) -> Optional[str]:
        """
        Send a chat request to LM Studio and return the response text.
        Uses the LRU cache to deduplicate identical prompts.
        Respects the circuit breaker.
        """
        if self._circuit_tripped:
            return None
        
        # Check LRU cache
        cache_key = self._hash_prompt(system_prompt, user_input)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        }).encode('utf-8')
        
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method='POST'
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                if isinstance(result, dict):
                    if 'choices' in result:
                        text = result['choices'][0].get('message', {}).get('content', '')
                    elif 'response' in result:
                        text = result['response']
                    elif 'content' in result:
                        content = result['content']
                        text = str(content) if not isinstance(content, str) else content
                    elif 'output' in result:
                        output = result['output']
                        text = str(output) if not isinstance(output, str) else output
                    else:
                        text = str(result)
                else:
                    text = str(result)
                
                # Cache the successful response
                self._cache.put(cache_key, text)
                self._record_success()
                return text
        except urllib.error.URLError as e:
            self._record_failure()
            print(f"[AlphaEvolve] LLM connection failed: {e}")
            return None
        except Exception as e:
            self._record_failure()
            print(f"[AlphaEvolve] LLM request error: {e}")
            return None
    
    def _parse_lambdas(self, response: str) -> Optional[Tuple[str, str]]:
        """Extract two lambda expressions from LLM response text."""
        if not response:
            return None
            
        if not isinstance(response, str):
            response = str(response)
        
        # Strip think tags if present (common with reasoning models)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Find all lambda expressions
        lambda_pattern = r'lambda\s+n\s*:\s*[^\n]+'
        lambdas = re.findall(lambda_pattern, response)
        
        if len(lambdas) >= 2:
            a_n = lambdas[0].strip().rstrip(',')
            b_n = lambdas[1].strip().rstrip(',')
            return a_n, b_n
        
        # Fallback: try line-by-line parsing
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        lambda_lines = [l for l in lines if 'lambda' in l]
        
        if len(lambda_lines) >= 2:
            for i, line in enumerate(lambda_lines[:2]):
                match = re.search(r'(lambda\s+n\s*:.+)', line)
                if match:
                    lambda_lines[i] = match.group(1).strip()
            return lambda_lines[0], lambda_lines[1]
        
        return None

    def propose_mutation(self, a_n_code: str, b_n_code: str, 
                         target_name: str, fitness: float) -> Optional[Tuple[str, str]]:
        """Ask the LLM to propose a mathematically-motivated mutation."""
        prompt = f"""The target mathematical constant is: {target_name}
Current program (fitness = {fitness:.2f} digits matched):
  a(n) = {a_n_code}
  b(n) = {b_n_code}

Propose a SINGLE mutation that might improve convergence. Output exactly 2 lines:
a(n) = lambda n: <expression>
b(n) = lambda n: <expression>"""
        
        result = self._chat(MUTATION_SYSTEM_PROMPT, prompt)
        return self._parse_lambdas(result)
    
    def propose_crossover(self, parent_a: dict, parent_b: dict,
                          target_name: str) -> Optional[Tuple[str, str]]:
        """Ask the LLM to intelligently combine two parent programs."""
        prompt = f"""Target constant: {target_name}

Parent A (fitness = {parent_a['fitness']:.2f}):
  a(n) = {parent_a['a_n']}
  b(n) = {parent_a['b_n']}

Parent B (fitness = {parent_b['fitness']:.2f}):
  a(n) = {parent_b['a_n']}
  b(n) = {parent_b['b_n']}

Create a child that combines the mathematical strengths of both parents.
Output exactly 2 lines:
a(n) = lambda n: <expression>
b(n) = lambda n: <expression>"""
        
        result = self._chat(CROSSOVER_SYSTEM_PROMPT, prompt)
        return self._parse_lambdas(result)
    
    def propose_novel(self, target_name: str, target_value: float,
                      archive_summary: str = "") -> Optional[Tuple[str, str]]:
        """Ask the LLM to propose a completely novel GCF program from scratch."""
        prompt = f"""Target constant: {target_name} ≈ {target_value:.15f}

{f"Previously explored (avoid duplicates): {archive_summary}" if archive_summary else ""}

Propose a novel Generalized Continued Fraction where a(n) and b(n) are defined by creative mathematical expressions. Think about:
- Quadratic, cubic, or quartic polynomials in n
- Products like n*(n+1) or (2*n+1)*(2*n+3)
- Alternating signs: (-1)**n
- Squares and cubes of linear expressions

Output exactly 2 lines:
a(n) = lambda n: <expression>
b(n) = lambda n: <expression>"""
        
        result = self._chat(MUTATION_SYSTEM_PROMPT, prompt)
        return self._parse_lambdas(result)

    # ──────────────────────────────────────────────────────────────────────
    #  Batch / Parallel API — eliminates sequential blocking
    # ──────────────────────────────────────────────────────────────────────

    def propose_mutations_parallel(self, parents: List[Dict],
                                   target_name: str) -> List[Optional[Tuple[str, str]]]:
        """
        Submit all mutation requests in parallel via ThreadPoolExecutor.
        
        The GIL releases during urllib I/O, so N concurrent requests 
        complete in ~1 round-trip time instead of N × round-trip time.
        Returns results in the same order as parents.
        """
        if not self.circuit_ok:
            return [None] * len(parents)
        
        executor = self._get_executor()
        futures = {}
        
        for i, parent in enumerate(parents):
            future = executor.submit(
                self.propose_mutation,
                parent['a_n'], parent['b_n'],
                target_name, parent['fitness']
            )
            futures[future] = i
        
        results = [None] * len(parents)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = None
        
        return results

    def propose_crossovers_parallel(self, parent_pairs: List[Tuple[Dict, Dict]],
                                    target_name: str) -> List[Optional[Tuple[str, str]]]:
        """Submit all crossover requests in parallel."""
        if not self.circuit_ok:
            return [None] * len(parent_pairs)
        
        executor = self._get_executor()
        futures = {}
        
        for i, (pa, pb) in enumerate(parent_pairs):
            future = executor.submit(
                self.propose_crossover, pa, pb, target_name
            )
            futures[future] = i
        
        results = [None] * len(parent_pairs)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = None
        
        return results
    
    @property
    def cache_stats(self) -> dict:
        """Return cache hit/miss statistics for monitoring."""
        return {
            'hits': self._cache.hits,
            'misses': self._cache.misses,
            'hit_rate': f"{self._cache.hit_rate:.1%}",
            'size': len(self._cache._cache),
        }


def random_mutation(a_n_code: str, b_n_code: str) -> Tuple[str, str]:
    """
    Fallback random mutation when LLM is unavailable.
    Applies simple structural transformations.
    """
    mutations = [
        # Coefficient perturbation
        lambda code: re.sub(r'(\d+)', lambda m: str(int(m.group()) + random.choice([-2,-1,1,2])), code, count=1),
        # Add a term
        lambda code: code.rstrip(')') + f" + {random.choice([1,-1])} * n" if 'lambda' in code else code,
        # Multiply by alternating sign
        lambda code: code.replace('lambda n:', 'lambda n: (-1)**n * (') + ')' if '(-1)**n' not in code else code,
        # Square a term
        lambda code: code.replace('n', 'n**2', 1) if 'n**2' not in code else code.replace('n**2', 'n', 1),
    ]
    
    mut_fn = random.choice(mutations)
    if random.random() < 0.5:
        return mut_fn(a_n_code), b_n_code
    else:
        return a_n_code, mut_fn(b_n_code)
