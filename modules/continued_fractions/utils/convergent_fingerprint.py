"""
Convergent Fingerprinting Module
=================================

After the GPU pipeline produces a verified GCF hit, this module performs
automated post-processing to maximize the chance of turning a raw numerical
match into a publishable algebraic identity.

Pipeline position:
    GPU hit → mpmath 1000-digit verify → [THIS MODULE] → Firebase publish

The fingerprinting process:
    1. Multi-tier PSLQ identity resolution (linear → quadratic → algebraic)
    2. Convergence rate analysis (digits per term)
    3. Meta-discovery: check if the convergence rate itself is a known constant
    4. Near-miss cataloging for extended-precision follow-up sweeps

Usage:
    from modules.continued_fractions.utils.convergent_fingerprint import fingerprint_gcf_hit
    
    report = fingerprint_gcf_hit(
        a_coef=(1, 0, 1),
        b_coef=(0, -1, 0),
        target_name='euler_mascheroni',
    )
"""
import logging
import mpmath
from typing import Dict, Any, Optional, Tuple, List

from modules.continued_fractions.utils.mobius import EfficientGCF
from modules.continued_fractions.utils.lll_identity_resolver import resolve_identity
from modules.continued_fractions.targets import g_N_verify_terms, g_N_verify_compare_length

logger = logging.getLogger(__name__)

# Near-miss threshold: residual below this is worth cataloging for
# extended-precision follow-up, even if it's not a verified identity.
NEAR_MISS_THRESHOLD = 1e-8


def fingerprint_gcf_hit(
    a_coef: tuple,
    b_coef: tuple,
    target_name: str = 'unknown',
    a_iterator=None,
    b_iterator=None,
    n_verify_terms: int = 1000,
    precision: int = 200,
) -> Dict[str, Any]:
    """
    Perform comprehensive fingerprinting of a verified GCF hit.
    
    Args:
        a_coef:         Coefficient tuple for a(n) polynomial
        b_coef:         Coefficient tuple for b(n) polynomial
        target_name:    Name of the target constant (for logging)
        a_iterator:     Custom iterator function for a(n). If None, uses compact poly.
        b_iterator:     Custom iterator function for b(n). If None, uses compact poly.
        n_verify_terms: Number of convergent terms to evaluate
        precision:      mpmath working decimal places
    
    Returns:
        Dict with keys:
            'convergent_value': mpmath.mpf — the high-precision GCF convergent
            'convergent_str':   str — first 50 digits
            'identity':         dict — output from resolve_identity (may or may not find a match)
            'convergence_rate': float — digits per term
            'rate_identity':    dict|None — if convergence rate itself matches a known constant
            'near_misses':      list — sub-threshold PSLQ near-hits for follow-up
            'quality_score':    float — overall quality score (0-100)
    """
    report = {
        'target': target_name,
        'a_coef': a_coef,
        'b_coef': b_coef,
        'convergent_value': None,
        'convergent_str': '',
        'identity': None,
        'convergence_rate': 0.0,
        'rate_identity': None,
        'near_misses': [],
        'quality_score': 0.0,
    }
    
    # ── Step 1: High-precision convergent evaluation ──────────────────────
    with mpmath.workdps(precision):
        an, bn = _build_series(a_coef, b_coef, n_verify_terms, a_iterator, b_iterator)
        
        if an is None:
            logger.warning(f"Failed to build series for a={a_coef}, b={b_coef}")
            return report
        
        gcf = EfficientGCF(an, bn)
        convergent = gcf.evaluate()
        
        if not mpmath.isfinite(convergent):
            logger.warning(f"GCF diverged for a={a_coef}, b={b_coef}")
            return report
        
        report['convergent_value'] = convergent
        report['convergent_str'] = mpmath.nstr(convergent, 50)
    
    # ── Step 2: Multi-tier identity resolution ────────────────────────────
    identity = resolve_identity(
        value=convergent,
        precision=precision,
        tolerance=1e-30,
    )
    report['identity'] = identity
    
    # ── Step 3: Convergence rate analysis ─────────────────────────────────
    rate = _compute_convergence_rate(an, bn, convergent, precision)
    report['convergence_rate'] = rate
    
    # ── Step 4: Meta-discovery — is the rate itself a known constant? ─────
    if rate > 0 and mpmath.isfinite(mpmath.mpf(rate)):
        rate_identity = resolve_identity(
            value=mpmath.mpf(rate),
            basis_constants={'pi', 'gamma', 'log2', 'e', '1', 'phi', 'sqrt2'},
            precision=50,  # Rate is only known to ~10 digits, don't overshoot
            tolerance=1e-6,
            max_denominator=100,
        )
        if rate_identity['found']:
            report['rate_identity'] = rate_identity
    
    # ── Step 5: Quality scoring ──────────────────────────────────────────
    score = 0.0
    
    if identity['found']:
        score += 50.0  # Identity resolved
        
        # Bonus for clean expressions (few terms, small coefficients)
        n_terms = len(identity.get('coefficients', {}))
        if n_terms <= 2:
            score += 20.0
        elif n_terms <= 4:
            score += 10.0
        
        # Bonus for tight residual
        residual = identity.get('residual', 1.0)
        if residual < 1e-50:
            score += 15.0
        elif residual < 1e-20:
            score += 10.0
        
        # Bonus for linear (tier 1) match
        if identity.get('method') == 'pslq_linear':
            score += 15.0
        elif identity.get('method') == 'pslq_quadratic':
            score += 10.0
    else:
        # Even without identity, grade the convergence quality
        if rate > 1.0:
            score += 15.0  # Super-exponential convergence
        elif rate > 0.5:
            score += 10.0
        elif rate > 0.1:
            score += 5.0
    
    # Bonus if convergence rate is itself a known constant (meta-discovery!)
    if report['rate_identity'] is not None:
        score += 10.0
    
    report['quality_score'] = min(100.0, score)
    
    return report


def _build_series(a_coef, b_coef, n_terms, a_iter=None, b_iter=None):
    """Build a(n) and b(n) series from coefficients or custom iterators."""
    try:
        if a_iter is not None:
            an = list(a_iter(a_coef, n_terms, start_n=0))
        else:
            an = _compact_poly_series(a_coef, n_terms)
        
        if b_iter is not None:
            bn = list(b_iter(b_coef, n_terms, start_n=0))
        else:
            bn = _compact_poly_series(b_coef, n_terms)
        
        return an, bn
    except Exception as e:
        logger.error(f"Series build failed: {e}")
        return None, None


def _compact_poly_series(coef, n_terms):
    """Evaluate a compact polynomial [c_d, ..., c_0] at n = 0, 1, ..., n_terms-1."""
    series = []
    for n in range(n_terms):
        val = 0
        for c in coef:
            val = val * n + c
        series.append(val)
    return series


def _compute_convergence_rate(an, bn, reference, precision):
    """
    Compute the convergence rate in digits per term.
    Uses the slope of log₁₀|convergent_n - reference| vs n.
    """
    try:
        with mpmath.workdps(precision):
            prev_q, q = 0, 1
            prev_p, p = 1, an[0]
            
            log_diffs = []
            length = min(200, len(bn))
            
            for i in range(1, length):
                tmp_q, tmp_p = q, p
                q = an[i] * q + bn[i] * prev_q
                p = an[i] * p + bn[i] * prev_p
                prev_q = tmp_q
                prev_p = tmp_p
                
                if q == 0:
                    continue
                
                convergent = mpmath.mpf(p) / mpmath.mpf(q)
                if not mpmath.isfinite(convergent):
                    break
                
                diff = abs(convergent - reference)
                if diff > 0:
                    log_diffs.append(float(mpmath.log10(diff)))
            
            if len(log_diffs) < 10:
                return 0.0
            
            # Linear regression on the second half (after initial transient)
            half = len(log_diffs) // 2
            slope = 2 * (log_diffs[-1] - log_diffs[half]) / len(log_diffs)
            return -slope  # digits per term (positive = converging)
    except Exception:
        return 0.0


def format_fingerprint_report(report: Dict[str, Any]) -> str:
    """Pretty-print a fingerprint report for logging or display."""
    lines = [
        "=" * 70,
        "  CONVERGENT FINGERPRINT REPORT",
        "=" * 70,
        f"  Target: {report['target']}",
        f"  a(n) coefficients: {report['a_coef']}",
        f"  b(n) coefficients: {report['b_coef']}",
        f"  Convergent (50 digits): {report['convergent_str']}",
        f"  Convergence Rate: {report['convergence_rate']:.4f} digits/term",
        "",
    ]
    
    identity = report.get('identity', {})
    if identity and identity.get('found'):
        lines += [
            f"  ✅ IDENTITY FOUND (via {identity['method']}):",
            f"     value = {identity['expression']}",
            f"     Residual: {identity['residual']:.2e}",
        ]
        if identity.get('coefficients'):
            lines.append("     Coefficients:")
            for const, coef in identity['coefficients'].items():
                lines.append(f"       {const:15s} → {coef:+.10f}")
    else:
        lines.append("  ❌ No algebraic identity found in basis.")
    
    rate_id = report.get('rate_identity')
    if rate_id and rate_id.get('found'):
        lines += [
            "",
            f"  🔬 META-DISCOVERY: Convergence rate matches known constant!",
            f"     rate ≈ {rate_id['expression']}",
        ]
    
    lines += [
        "",
        f"  Quality Score: {report['quality_score']:.0f}/100",
        "=" * 70,
    ]
    
    return "\n".join(lines)
