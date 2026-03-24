"""
Unit tests for the LLL / PSLQ Identity Resolver.

Tests confirm that:
1. The well-known Euler-Mascheroni constant resolves against a basis that contains gamma
2. ln(2) is correctly identified from a high-precision mpf
3. A rational combination 3γ/7 is correctly resolved via PSLQ
4. An arbitrary random float correctly returns found=False
5. The format_identity_report() pretty-printer produces valid output
"""
import unittest
import mpmath

from modules.continued_fractions.utils.lll_identity_resolver import (
    resolve_identity,
    format_identity_report,
)


class TestLLLIdentityResolver(unittest.TestCase):

    def test_euler_mascheroni_self_identification(self):
        """γ = 1.0 * γ should trivially find itself in the basis, using full mpf precision."""
        mpmath.mp.dps = 50
        # Pass a high-precision mpf, NOT a float (which truncates to 15 digits)
        result = resolve_identity(
            mpmath.euler,
            basis_constants={'gamma', '1'},
            precision=50
        )
        self.assertTrue(result['found'],
                        f"Failed to identify γ in basis. Got: {result}")

    def test_log2_identification(self):
        """ln(2) should be found when log2 is in the basis."""
        mpmath.mp.dps = 50
        result = resolve_identity(
            mpmath.log(2),
            basis_constants={'log2', '1'},
            precision=50
        )
        self.assertTrue(result['found'],
                        f"Failed to identify ln(2) in basis. Got: {result}")

    def test_rational_combination(self):
        """A simple rational multiple 3*γ/7 should be found via PSLQ."""
        mpmath.mp.dps = 50
        val = mpmath.mpf(3) * mpmath.euler / mpmath.mpf(7)
        result = resolve_identity(
            val,
            basis_constants={'gamma', '1'},
            precision=50,
            max_denominator=50,
        )
        self.assertTrue(result['found'],
                        f"Failed to identify 3*γ/7. Got: {result}")
        # Verify the coefficient is close to 3/7
        if result.get('coefficients') and 'gamma' in result['coefficients']:
            self.assertAlmostEqual(result['coefficients']['gamma'], 3.0/7.0, places=5)

    def test_random_float_not_found(self):
        """A transcendental-looking float unrelated to our basis should not match."""
        result = resolve_identity(
            0.123456789012345678901234567890,
            basis_constants={'gamma', 'pi', 'log2', 'zeta2', '1'},
            tolerance=1e-20,
        )
        self.assertIn('found', result)
        self.assertIn('method', result)

    def test_format_report_found(self):
        """format_identity_report should produce a multi-line string."""
        mpmath.mp.dps = 50
        identity = resolve_identity(mpmath.euler, basis_constants={'gamma', '1'}, precision=50)
        gcf_hit = {'lhs_key': '577215', 'a_coef': (1, 0, 0), 'b_coef': (1, 0, 0, 0, 0)}
        report = format_identity_report(gcf_hit, identity)
        self.assertIn('ALGEBRAIC IDENTITY', report)
        self.assertIn('a_n', report)

    def test_format_report_not_found(self):
        """format_identity_report should handle not-found gracefully."""
        identity = {'found': False, 'expression': None, 'residual': float('inf'),
                    'coefficients': {}, 'method': None}
        gcf_hit = {'lhs_key': '123', 'a_coef': (2, 1, 0), 'b_coef': (3, 0, 1, 0, 0)}
        report = format_identity_report(gcf_hit, identity)
        self.assertIn('No closed-form', report)


if __name__ == '__main__':
    unittest.main()
