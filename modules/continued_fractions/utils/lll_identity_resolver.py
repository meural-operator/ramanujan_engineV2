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
import logging
import mpmath
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


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
        'gamma':   mpmath.euler,          # Euler-Mascheroni γ ≈ 0.5772...
        'pi':      mpmath.pi,             # π
        'log2':    mpmath.log(2),          # ln(2)
        'log3':    mpmath.log(3),          # ln(3)
        'zeta2':   mpmath.zeta(2),         # π²/6
        'zeta3':   mpmath.apery,           # Apéry's constant ζ(3)
        'zeta5':   mpmath.zeta(5),         # ζ(5) — primary GPU campaign target
        'zeta7':   mpmath.zeta(7),         # ζ(7) — primary GPU campaign target
        'catalan': mpmath.catalan,         # Catalan's constant G ≈ 0.9159...
        'sqrt2':   mpmath.sqrt(2),         # √2
        'sqrt3':   mpmath.sqrt(3),         # √3
        'sqrt5':   mpmath.sqrt(5),         # √5
        'e':       mpmath.e,               # Euler's number e
        'phi':     (1 + mpmath.sqrt(5))/2, # Golden ratio φ
        '1':       mpmath.mpf(1),          # rational intercept anchor
    }
    return {k: v for k, v in _registry.items() if k in names}


def _build_quadratic_basis(names: set, precision: int) -> Dict[str, Any]:
    """
    Extend the linear basis with products and squares of constants.
    
    This allows PSLQ to find relations like:
        v = (1/6) * pi^2          (= ζ(2))
        v = (7/180) * pi^4        (= ζ(4)) 
        v = gamma * pi + 3*log2
    
    We only include products of the most commonly appearing constants
    to keep the basis size manageable (PSLQ reliability degrades with
    basis size > ~20 elements at 200 dps).
    """
    linear = _build_basis(names, precision)
    quadratic = dict(linear)  # start with linear basis
    
    # Core constants that appear in products in known identities
    product_candidates = ['pi', 'gamma', 'log2', 'e', 'zeta3']
    available = [c for c in product_candidates if c in linear]
    
    # Add squares: π², γ², log²(2), etc.
    for c in available:
        key = f"{c}^2"
        quadratic[key] = linear[c] ** 2
    
    # Add pairwise products: π·γ, π·log(2), γ·log(2), etc.
    for idx_i in range(len(available)):
        for idx_j in range(idx_i + 1, len(available)):
            c1, c2 = available[idx_i], available[idx_j]
            key = f"{c1}*{c2}"
            quadratic[key] = linear[c1] * linear[c2]
    
    # Add specific high-value composites from the literature
    mpmath.mp.dps = precision
    if 'pi' in linear:
        quadratic['pi^4'] = linear['pi'] ** 4  # appears in ζ(4) = π⁴/90
    if 'pi' in linear and 'gamma' in linear:
        quadratic['gamma/pi'] = linear['gamma'] / linear['pi']
    
    return quadratic


def _build_algebraic_basis(names: set, precision: int) -> Dict[str, Any]:
    """
    Extend the linear basis with algebraic (root) expressions.
    
    This allows PSLQ to find relations involving:
        √π, ∛2, π^(1/3), etc.
    
    These appear in some Ramanujan-type formulas.
    Kept small to avoid PSLQ unreliability.
    """
    linear = _build_basis(names, precision)
    algebraic = dict(linear)
    
    mpmath.mp.dps = precision
    
    # Square roots of core constants
    root_candidates = ['pi', 'e', 'zeta3', 'zeta5']
    for c in root_candidates:
        if c in linear and linear[c] > 0:
            algebraic[f"sqrt({c})"] = mpmath.sqrt(linear[c])
    
    # Cube roots (appear in some elliptic integral formulas)
    for c in ['pi', 'e']:
        if c in linear and linear[c] > 0:
            algebraic[f"cbrt({c})"] = mpmath.cbrt(linear[c])
    
    # Specific algebraic expressions from Ramanujan's work
    if 'pi' in linear:
        algebraic['1/pi'] = 1 / linear['pi']
        algebraic['pi^(3/2)'] = linear['pi'] ** mpmath.mpf('1.5')
    
    return algebraic


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
                            Defaults to a comprehensive basis covering all
                            implemented search domains.
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
        # Full basis covering all implemented search domains
        basis_constants = {
            'gamma', 'pi', 'log2', 'log3',
            'zeta2', 'zeta3', 'zeta5', 'zeta7',
            'catalan', '1',
        }

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
            safe_env = _mpmath_safe_env()
            try:
                reconstructed = eval(ident, {'__builtins__': {}}, safe_env)
                residual = float(abs(val - mpmath.mpf(reconstructed)))
                result['found'] = True
                result['expression'] = str(ident)
                result['method'] = 'mpmath_identify'
                result['residual'] = residual
                return result
            except NameError as e:
                # mpmath.identify() returned a token not in our safe_env
                logger.warning(
                    f"mpmath.identify() returned expression '{ident}' containing "
                    f"unknown token: {e}. Expression discarded — consider extending "
                    f"_mpmath_safe_env() with the missing function/constant."
                )
            except SyntaxError as e:
                logger.warning(
                    f"mpmath.identify() returned unparseable expression '{ident}': {e}"
                )
            except (TypeError, ValueError, ZeroDivisionError) as e:
                logger.warning(
                    f"mpmath.identify() expression '{ident}' failed evaluation: {e}"
                )
    except Exception:
        pass

    # ── Method 2: Graded PSLQ Search ─────────────────────────────────────────
    # Tier 1: Linear basis (fast, small vector, high reliability)
    # Tier 2: Quadratic basis (products & squares of constants)
    # Tier 3: Algebraic basis (square roots & cube roots)
    #
    # Each tier only runs if the previous one failed.
    # Larger bases need higher precision to avoid spurious relations.
    
    tiers = [
        ('pslq_linear',    _build_basis(basis_constants, precision), precision),
        ('pslq_quadratic', _build_quadratic_basis(basis_constants, precision + 50), precision + 50),
        ('pslq_algebraic', _build_algebraic_basis(basis_constants, precision + 100), precision + 100),
    ]
    
    for method_name, basis, tier_precision in tiers:
        if not basis:
            continue
        
        # Safety guard: PSLQ with too many basis elements relative to precision
        # can find spurious integer relations. Rule of thumb: need at least
        # ~8 digits of working precision per basis element for PSLQ reliability.
        max_safe_basis_size = tier_precision // 8
        if len(basis) > max_safe_basis_size:
            logger.warning(
                f"Skipping {method_name}: basis size {len(basis)} exceeds "
                f"safe limit {max_safe_basis_size} for {tier_precision} dps"
            )
            continue
        
        mpmath.mp.dps = tier_precision
        names = list(basis.keys())
        vector = [val] + [basis[n] for n in names]
        
        try:
            relation = mpmath.pslq(vector, maxcoeff=max_denominator, tol=tolerance)
            if relation is not None and relation[0] != 0:
                n0 = int(relation[0])
                terms = {}
                parts = []
                for idx, name in enumerate(names):
                    ni = int(relation[idx + 1])
                    if ni != 0:
                        terms[name] = mpmath.mpf(-ni) / mpmath.mpf(n0)
                        coef_str = f"{-ni}/{n0}"
                        parts.append(f"({coef_str}) * {name}")
    
                if parts:
                    expr_str = " + ".join(parts)
                    reconstructed = sum(terms[n] * basis[n] for n in terms)
                    residual = float(abs(val - reconstructed))
    
                    if residual < 1e-10:
                        result['found'] = True
                        result['expression'] = expr_str
                        result['coefficients'] = {n: float(terms[n]) for n in terms}
                        result['residual'] = residual
                        result['method'] = method_name
                        return result  # Found — stop searching deeper tiers
    
        except Exception as e:
            logger.debug(f"{method_name} failed: {e}")
            continue

    return result


def _mpmath_safe_env() -> dict:
    """
    Safe evaluation namespace for mpmath.identify() string output.
    
    mpmath.identify() can return expressions involving any of mpmath's built-in
    constants and functions. This dict must cover all plausible tokens to prevent
    silent NameError discards.
    """
    return {
        # Constants
        'pi': mpmath.pi,
        'e': mpmath.e,
        'euler': mpmath.euler,
        'gamma': mpmath.euler,
        'apery': mpmath.apery,
        'catalan': mpmath.catalan,
        'khinchin': mpmath.khinchin,
        'glaisher': mpmath.glaisher,
        'mertens': mpmath.mertens,
        'twinprime': mpmath.twinprime,
        'phi': (1 + mpmath.sqrt(5)) / 2,
        
        # Functions commonly appearing in identify() output
        'log': mpmath.log,
        'sqrt': mpmath.sqrt,
        'cbrt': mpmath.cbrt,
        'exp': mpmath.exp,
        'zeta': mpmath.zeta,
        'power': mpmath.power,
        'root': mpmath.root,
        'cos': mpmath.cos,
        'sin': mpmath.sin,
        'tan': mpmath.tan,
        'atan': mpmath.atan,
        'frac': mpmath.frac,
        'mpf': mpmath.mpf,
        'mpc': mpmath.mpc,
        
        # Arithmetic helpers that identify() may emit
        'sign': mpmath.sign,
        'fabs': mpmath.fabs,
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
            f"    value = {identity['expression']}",
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
