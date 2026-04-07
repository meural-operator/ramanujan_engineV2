"""
Program Sandbox — Safe execution environment for LLM-generated GCF programs.

Provides restricted evaluation of Python lambda expressions with:
- Whitelisted mathematical operations only
- Timeout protection via multiprocessing (properly kills hung evaluations)
- NaN/Inf/exception detection and automatic disqualification
- Sequence generation for a_n and b_n evaluation
"""
import math
import multiprocessing
from typing import Optional, List, Callable, Tuple

# Whitelisted builtins available to evolved programs
SAFE_GLOBALS = {
    "__builtins__": {},
    "abs": abs,
    "min": min,
    "max": max,
    "int": int,
    "float": float,
    "round": round,
    "pow": pow,
    "divmod": divmod,
    "True": True,
    "False": False,
}

# Blacklisted tokens that must NEVER appear in program code
BLACKLISTED_TOKENS = [
    'import', 'exec', 'eval', 'open', 'os.', 'sys.', '__', 
    'subprocess', 'shutil', 'glob', 'pathlib', 'socket',
    'compile', 'getattr', 'setattr', 'delattr', 'globals', 'locals',
    'breakpoint', 'exit', 'quit', 'input',
]


def is_safe(code: str) -> bool:
    """Check if a lambda expression string is safe to evaluate."""
    code_lower = code.lower()
    for token in BLACKLISTED_TOKENS:
        if token in code_lower:
            return False
    
    # Must be a lambda expression
    if 'lambda' not in code:
        return False
    
    return True


def compile_lambda(code: str) -> Optional[Callable]:
    """
    Safely compile a lambda expression string into a callable.
    Returns None if compilation fails or code is unsafe.
    """
    if not is_safe(code):
        return None
    
    try:
        # Compile in restricted namespace
        func = eval(code, SAFE_GLOBALS)
        if not callable(func):
            return None
        return func
    except Exception:
        return None


def evaluate_sequence(func: Callable, n_terms: int) -> Optional[List[float]]:
    """
    Generate a sequence [func(0), func(1), ..., func(n_terms-1)].
    Returns None if any evaluation produces NaN, Inf, or raises an exception.
    """
    sequence = []
    for n in range(n_terms):
        try:
            val = float(func(n))
            if math.isnan(val) or math.isinf(val):
                return None
            if abs(val) > 1e15:  # Prevent overflow cascades
                return None
            sequence.append(val)
        except (ZeroDivisionError, OverflowError, ValueError, TypeError, RecursionError):
            return None
    return sequence


def _eval_worker(a_n_code, b_n_code, target_value, n_terms, result_queue):
    """
    Worker process for timeout-protected GCF evaluation.
    Runs in a separate process so it can be forcibly terminated if it hangs
    in a tight C-extension loop (numpy, mpmath, etc).
    """
    try:
        result = _compute_fitness(a_n_code, b_n_code, target_value, n_terms)
        result_queue.put(result)
    except Exception as e:
        result_queue.put({'valid': False, 'fitness': 0.0, 'convergence_rate': 0.0,
                          'value': None, 'error': str(e)})


def _compute_fitness(a_n_code, b_n_code, target_value, n_terms):
    """Core fitness computation (runs inside worker process)."""
    a_func = compile_lambda(a_n_code)
    b_func = compile_lambda(b_n_code)
    
    if a_func is None:
        return {'valid': False, 'fitness': 0.0, 'convergence_rate': 0.0,
                'value': None, 'error': f'Failed to compile a(n): {a_n_code}'}
    if b_func is None:
        return {'valid': False, 'fitness': 0.0, 'convergence_rate': 0.0,
                'value': None, 'error': f'Failed to compile b(n): {b_n_code}'}
    
    a_seq = evaluate_sequence(a_func, n_terms)
    b_seq = evaluate_sequence(b_func, n_terms)
    
    if a_seq is None or b_seq is None:
        return {'valid': False, 'fitness': 0.0, 'convergence_rate': 0.0,
                'value': None, 'error': 'Sequence generation failed (NaN/Inf/overflow)'}
    
    if len(a_seq) < 2 or len(b_seq) < 2:
        return {'valid': False, 'fitness': 0.0, 'convergence_rate': 0.0,
                'value': None, 'error': 'Sequence too short'}
    
    # Standard GCF recurrence
    prev_p = 1.0
    p = a_seq[0]
    prev_q = 0.0
    q = 1.0
    
    convergent_history = []
    
    for i in range(1, min(len(a_seq), len(b_seq))):
        a_i = a_seq[i]
        b_i = b_seq[i]
        
        new_p = a_i * p + b_i * prev_p
        new_q = a_i * q + b_i * prev_q
        prev_p = p
        prev_q = q
        p = new_p
        q = new_q
        
        # Periodic scaling to prevent overflow
        if abs(q) > 1e10:
            scale = abs(q)
            p /= scale
            q /= scale
            prev_p /= scale
            prev_q /= scale
        
        if q != 0 and not math.isnan(p/q) and not math.isinf(p/q):
            convergent_history.append(p / q)
    
    if not convergent_history:
        return {'valid': False, 'fitness': 0.0, 'convergence_rate': 0.0,
                'value': None, 'error': 'No valid convergents produced'}
    
    final_value = convergent_history[-1]
    
    if math.isnan(final_value) or math.isinf(final_value):
        return {'valid': False, 'fitness': 0.0, 'convergence_rate': 0.0,
                'value': None, 'error': 'Final convergent is NaN/Inf'}
    
    # Fitness = digits of accuracy
    error = abs(final_value - target_value)
    if error == 0:
        digits = 15.0
    elif error > abs(target_value) * 10:
        digits = 0.0
    else:
        try:
            digits = max(0.0, -math.log10(error + 1e-300))
        except (ValueError, OverflowError):
            digits = 0.0
    
    # Convergence rate
    if len(convergent_history) >= 10:
        early_error = abs(convergent_history[len(convergent_history)//4] - target_value)
        late_error = abs(convergent_history[-1] - target_value)
        if early_error > 0 and late_error > 0:
            try:
                rate = math.log10(early_error / (late_error + 1e-300)) / max(1, len(convergent_history))
                rate = max(0.0, min(1.0, rate))
            except (ValueError, OverflowError):
                rate = 0.0
        else:
            rate = 1.0 if late_error == 0 else 0.0
    else:
        rate = 0.0
    
    return {
        'valid': True,
        'fitness': digits,
        'convergence_rate': rate,
        'value': final_value,
        'error': None
    }


def evaluate_gcf_fitness(a_n_code: str, b_n_code: str, target_value: float, 
                         n_terms: int = 200, timeout_sec: float = 2.0) -> dict:
    """
    Evaluate a GCF program's fitness against a target mathematical constant.
    
    Uses multiprocessing for proper timeout enforcement — if the evaluation
    hangs in a tight C-extension loop (numpy, mpmath), the worker process
    is forcibly terminated via Process.terminate(). This is a hard kill that
    actually reclaims resources, unlike threading.Thread.join(timeout) which
    leaves zombie threads running indefinitely.
    
    Args:
        a_n_code: Lambda string for the a(n) sequence
        b_n_code: Lambda string for the b(n) sequence
        target_value: The mathematical constant to converge to
        n_terms: Number of GCF terms to evaluate
        timeout_sec: Maximum seconds allowed for evaluation
        
    Returns:
        Dict with keys: 'valid', 'fitness', 'convergence_rate', 'value', 'error'
    """
    result = {
        'valid': False, 
        'fitness': 0.0, 
        'convergence_rate': 0.0,
        'value': None, 
        'error': None
    }
    
    # Quick safety check before spawning a process
    if not is_safe(a_n_code) or not is_safe(b_n_code):
        result['error'] = 'Unsafe code detected'
        return result
    
    result_queue = multiprocessing.Queue()
    
    proc = multiprocessing.Process(
        target=_eval_worker,
        args=(a_n_code, b_n_code, target_value, n_terms, result_queue)
    )
    proc.start()
    proc.join(timeout=timeout_sec)
    
    if proc.is_alive():
        # Forcibly kill the hung process — this is the key advantage over threading
        proc.terminate()
        proc.join(timeout=1.0)  # Give it a moment to clean up
        if proc.is_alive():
            proc.kill()  # SIGKILL as last resort
        result['error'] = 'Evaluation timed out (process killed)'
        return result
    
    try:
        if not result_queue.empty():
            return result_queue.get_nowait()
    except Exception:
        pass
    
    result['error'] = 'Worker process returned no result'
    return result
