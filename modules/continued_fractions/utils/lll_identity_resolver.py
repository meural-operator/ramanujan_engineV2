"""
LLL-Based Integer Relation / Closed-Form Identity Resolver
===========================================================

After the GPU CUDA sweep finds a raw float hit and the CPU mpmath verifier
confirms it at high precision, this module applies the LLL (Lenstra-Lenstra-Lovász)
lattice basis reduction algorithm to ask:

    "Can this number be expressed as an exact rational linear combination
     of known mathematical constants?"

This is how the original Ramanujan Machine (Technion, 2019) turned raw numerical
hits into formally publishable algebraic identities like:
    γ = a/b * (something with π, ζ(2), log(2), ...)

Pipeline position:
    GPU hit → mpmath 1000-digit verify → [THIS MODULE] → Firebase publish

Dependencies (ordered by preference):
  1. mpmath.identify() — PSLQ-based, built into mpmath, no extra install
  2. fpylll — Python bindings for the fplll C library (pip install fpylll)

Usage:
    from modules.continued_fractions.utils.lll_identity_resolver import resolve_identity
    result = resolve_identity(decimal_value, basis_constants={'gamma', 'pi', 'log2', 'zeta2'})
"""
import mpmath
from typing import Optional, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# Basis constant registry
# Extend this dict to widen the search space for future constants
# ─────────────────────────────────────────────────────────────────────────────
def _build_basis(names: set, precision: int) -> Dict[str, Any]:
    """
    Construct a dict of high-precision mpmath constant evaluations.
    Only includes constants requested in `names`.
    """
    mpmath.mp.dps = precision
    _registry = {
        'gamma':  mpmath.euler,          # Euler-Mascheroni γ ≈ 0.5772...
        'pi':     mpmath.pi,             # π
        'log2':   mpmath.log(2),         # ln(2)
        'log3':   mpmath.log(3),         # ln(3)
        'zeta2':  mpmath.zeta(2),        # π²/6
        'zeta3':  mpmath.apery,          # Apéry's constant ζ(3)
        'sqrt2':  mpmath.sqrt(2),        # √2
        'sqrt5':  mpmath.sqrt(5),        # √5
        'e':      mpmath.e,              # Euler's number e
        '1':      mpmath.mpf(1),         # rational intercept anchor
    }
    return {k: v for k, v in _registry.items() if k in names}


def resolve_identity(
    value: float,
    basis_constants: Optional[set] = None,
    max_denominator: int = 1000,
    precision: int = 150,
    tolerance: float = 1e-30,
) -> Dict[str, Any]:
    """
    Apply PSLQ/LLL integer relation search to find a closed-form algebraic
    expression for `value` as a rational linear combination of basis constants.

    Args:
        value:              The confirmed high-precision GCF output (float or mpf)
        basis_constants:    Set of constant names to search over.
                            Defaults to the full Euler-Mascheroni relevant basis:
                            {gamma, pi, log2, zeta2, zeta3, 1}
        max_denominator:    Maximum integer coefficient magnitude to try
        precision:          mpmath decimal precision for internal arithmetic
        tolerance:          Residual error threshold for accepting a match

    Returns:
        A dict with:
          'found'      : bool — whether an exact identity was found
          'expression' : str  — human-readable algebraic identity string (if found)
          'residual'   : float — |value - reconstructed_expression|
          'coefficients': dict — mapping constant_name → rational coefficient
          'method'     : str  — which solver found the match ('pslq' or 'mpmath_identify')
    """
    if basis_constants is None:
        basis_constants = {'gamma', 'pi', 'log2', 'zeta2', '1'}

    mpmath.mp.dps = precision
    # Preserve full precision if caller passes an mpf; float inputs will be limited to ~15 digits
    val = value if isinstance(value, mpmath.mpf) else mpmath.mpf(value)

    result = {
        'found': False,
        'expression': None,
        'residual': float('inf'),
        'coefficients': {},
        'method': None,
    }

    # ── Method 1: mpmath.identify() ──────────────────────────────────────────
    # Fastest — tries π, γ, e combinatorics via internal PSLQ-based search
    try:
        ident = mpmath.identify(val, tol=tolerance)
        if ident:
            result['found'] = True
            result['expression'] = str(ident)
            result['method'] = 'mpmath_identify'
            result['residual'] = float(abs(val - mpmath.mpf(eval(ident, {'__builtins__': {}},
                                                                  _mpmath_safe_env()))))
            return result
    except Exception:
        pass

    # ── Method 2: Explicit rational PSLQ over a chosen basis ─────────────────
    # Constructs a numerical vector [value, c1, c2, c3, ...] and finds
    # integer relation [n0, n1, n2, ...] such that sum(ni * vi) = 0
    basis = _build_basis(basis_constants, precision)
    names = list(basis.keys())
    vector = [val] + [basis[n] for n in names]

    try:
        relation = mpmath.pslq(vector, maxcoeff=max_denominator, tol=tolerance)
        if relation is not None and relation[0] != 0:
            # Relation: relation[0]*value + relation[1]*c1 + ... = 0
            # Rearranged: value = -(relation[1]*c1 + ...) / relation[0]
            n0 = int(relation[0])
            terms = {}
            parts = []
            for i, name in enumerate(names):
                ni = int(relation[i + 1])
                if ni != 0:
                    terms[name] = mpmath.mpf(-ni) / mpmath.mpf(n0)
                    coef_str = f"{-ni}/{n0}"
                    parts.append(f"({coef_str}) * {name}")

            if parts:
                expr_str = " + ".join(parts)
                reconstructed = sum(terms[n] * basis[n] for n in terms)
                residual = float(abs(val - reconstructed))

                # Accept if residual is small relative to precision
                if residual < 1e-10:
                    result['found'] = True
                    result['expression'] = expr_str
                    result['coefficients'] = {n: float(terms[n]) for n in terms}
                    result['residual'] = residual
                    result['method'] = 'pslq_basis'

    except Exception as e:
        result['error'] = str(e)

    return result


def _mpmath_safe_env() -> dict:
    """Safe evaluation namespace for mpmath.identify() string output."""
    return {
        'pi': mpmath.pi,
        'e': mpmath.e,
        'euler': mpmath.euler,
        'log': mpmath.log,
        'sqrt': mpmath.sqrt,
        'zeta': mpmath.zeta,
        'apery': mpmath.apery,
        'gamma': mpmath.euler,
    }


def format_identity_report(gcf_hit: dict, identity: dict) -> str:
    """
    Pretty-print a full discovery report combining the GCF polynomial
    coefficients and the resolved algebraic identity.

    Args:
        gcf_hit:  Dictionary from the GPU pipeline with keys: lhs_key, a_coef, b_coef
        identity: Output dict from resolve_identity()

    Returns:
        A formatted multi-line string suitable for logging or Firebase upload.
    """
    lines = [
        "=" * 60,
        "  *** ALGEBRAIC IDENTITY CANDIDATE DISCOVERED ***",
        "=" * 60,
        f"  GCF Polynomial a_n: {gcf_hit.get('a_coef')}",
        f"  GCF Polynomial b_n: {gcf_hit.get('b_coef')}",
        f"  LHS Key (hash):     {gcf_hit.get('lhs_key')}",
        "",
    ]

    if identity['found']:
        lines += [
            f"  Closed-Form Identity (via {identity['method']}):",
            f"    γ = {identity['expression']}",
            f"  Verification Residual: {identity['residual']:.2e}",
        ]
        if identity['coefficients']:
            lines.append("  Rational Coefficients:")
            for const, coef in identity['coefficients'].items():
                lines.append(f"    {const:12s}  →  {coef:+.10f}")
    else:
        lines.append("  [!] No closed-form algebraic identity found in basis.")
        lines.append("      Consider expanding the basis_constants set.")

    lines.append("=" * 60)
    return "\n".join(lines)
