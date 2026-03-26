"""
Program Sandbox — Safe execution environment for LLM-generated GCF programs.

Provides restricted evaluation of Python lambda expressions with:
- Whitelisted mathematical operations only
- Timeout protection via threading
- NaN/Inf/exception detection and automatic disqualification
- Sequence generation for a_n and b_n evaluation
"""
import math
import signal
import threading
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


def _eval_with_timeout(func, args, timeout_sec, result_holder):
    """Worker thread for timeout-protected evaluation."""
    try:
        result_holder['value'] = func(*args)
    except Exception as e:
        result_holder['error'] = str(e)


def evaluate_gcf_fitness(a_n_code: str, b_n_code: str, target_value: float, 
                         n_terms: int = 200, timeout_sec: float = 2.0) -> dict:
    """
    Evaluate a GCF program's fitness against a target mathematical constant.
    
    The fitness is measured as the number of digits of accuracy achieved
    by the GCF convergent after n_terms iterations.
    
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
    
    # 1. Compile lambdas safely
    a_func = compile_lambda(a_n_code)
    b_func = compile_lambda(b_n_code)
    
    if a_func is None:
        result['error'] = f'Failed to compile a(n): {a_n_code}'
        return result
    if b_func is None:
        result['error'] = f'Failed to compile b(n): {b_n_code}'
        return result
    
    # 2. Generate sequences with timeout protection
    result_holder = {'value': None, 'error': None}
    
    def _compute():
        a_seq = evaluate_sequence(a_func, n_terms)
        b_seq = evaluate_sequence(b_func, n_terms)
        
        if a_seq is None or b_seq is None:
            return {'valid': False, 'error': 'Sequence generation failed (NaN/Inf/overflow)'}
        
        # 3. Evaluate GCF using standard recurrence relation
        # p_{-1} = 1, p_0 = a_0
        # q_{-1} = 0, q_0 = 1
        # p_n = a_n * p_{n-1} + b_n * p_{n-2}
        # q_n = a_n * q_{n-1} + b_n * q_{n-2}
        if len(a_seq) < 2 or len(b_seq) < 2:
            return {'valid': False, 'error': 'Sequence too short'}
        
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
            return {'valid': False, 'error': 'No valid convergents produced'}
        
        final_value = convergent_history[-1]
        
        if math.isnan(final_value) or math.isinf(final_value):
            return {'valid': False, 'error': 'Final convergent is NaN/Inf'}
        
        # 4. Calculate fitness = digits of accuracy
        error = abs(final_value - target_value)
        if error == 0:
            digits = 15.0  # float64 max precision
        elif error > abs(target_value) * 10:
            digits = 0.0  # Completely wrong
        else:
            try:
                digits = max(0.0, -math.log10(error + 1e-300))
            except (ValueError, OverflowError):
                digits = 0.0
        
        # 5. Calculate convergence rate (how fast it approaches)
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
    
    # Run with timeout
    thread = threading.Thread(target=lambda: result_holder.update({'value': _compute()}))
    thread.start()
    thread.join(timeout=timeout_sec)
    
    if thread.is_alive():
        result['error'] = 'Evaluation timed out'
        return result
    
    if result_holder.get('value'):
        return result_holder['value']
    
    result['error'] = result_holder.get('error', 'Unknown evaluation error')
    return result
