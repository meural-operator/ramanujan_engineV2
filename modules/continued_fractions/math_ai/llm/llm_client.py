"""
LM Studio Client for AlphaEvolve — LLM-Guided Evolutionary GCF Discovery.

Connects to a local LM Studio instance running Qwen3-Coder-30B to propose
mathematically-motivated mutations and crossovers for GCF program evolution.
"""
import json
import re
import random
import urllib.request
import urllib.error
from typing import Optional, Tuple

# Default LM Studio configuration
DEFAULT_BASE_URL = "http://127.0.0.1:1234"
DEFAULT_MODEL = "qwen/qwen3-coder-30b"

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


class LMStudioClient:
    """
    Lightweight client for LM Studio's local inference API.
    Uses only stdlib (urllib) to avoid adding dependencies.
    """
    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 timeout: int = 60):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self._available = None
    
    def is_available(self) -> bool:
        """Check if LM Studio is reachable."""
        if self._available is not None:
            return self._available
        try:
            req = urllib.request.Request(f"{self.base_url}/api/v1/models", method='GET')
            urllib.request.urlopen(req, timeout=5)
            self._available = True
        except Exception:
            self._available = False
        return self._available
    
    def _chat(self, system_prompt: str, user_input: str) -> Optional[str]:
        """Send a chat request to LM Studio and return the response text."""
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
                # LM Studio response format — extract the text
                if isinstance(result, dict):
                    # Try common response formats
                    if 'choices' in result:
                        return result['choices'][0].get('message', {}).get('content', '')
                    elif 'response' in result:
                        return result['response']
                    elif 'content' in result:
                        content = result['content']
                        return str(content) if not isinstance(content, str) else content
                    elif 'output' in result:
                        output = result['output']
                        return str(output) if not isinstance(output, str) else output
                    else:
                        # Return the whole thing as string for debugging
                        return str(result)
                return str(result)
        except urllib.error.URLError as e:
            print(f"[AlphaEvolve] LLM connection failed: {e}")
            return None
        except Exception as e:
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
        # Filter to lines containing 'lambda'
        lambda_lines = [l for l in lines if 'lambda' in l]
        
        if len(lambda_lines) >= 2:
            # Extract just the lambda part 
            for i, line in enumerate(lambda_lines[:2]):
                # Remove prefixes like "a(n) = " or "1. "
                match = re.search(r'(lambda\s+n\s*:.+)', line)
                if match:
                    lambda_lines[i] = match.group(1).strip()
            return lambda_lines[0], lambda_lines[1]
        
        return None

    def propose_mutation(self, a_n_code: str, b_n_code: str, 
                         target_name: str, fitness: float) -> Optional[Tuple[str, str]]:
        """
        Ask the LLM to propose a mathematically-motivated mutation of the given program.
        
        Returns:
            Tuple of (mutated_a_n, mutated_b_n) lambda strings, or None if LLM fails.
        """
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
        """
        Ask the LLM to intelligently combine two parent programs.
        """
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
        """
        Ask the LLM to propose a completely novel GCF program from scratch.
        """
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
    # Randomly choose which to mutate
    if random.random() < 0.5:
        return mut_fn(a_n_code), b_n_code
    else:
        return a_n_code, mut_fn(b_n_code)
