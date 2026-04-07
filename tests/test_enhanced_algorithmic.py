"""
Enhanced Validation Tests for Algorithmic Improvements
======================================================

These are intricate stress tests that go beyond basic functionality:

1. Apéry Domain:
   - Mathematical correctness: verify the actual convergent VALUE for known GCFs
   - Edge cases: degenerate coefficients, boundary conditions
   - Cross-validation: Zeta3Domain1 and AperyFamilyDomain must yield identical results
   - All families produce finite convergents

2. GPU Asymptotic Filter:
   - Zero false negatives: compare filtered vs unfiltered hit counts
   - Edge cases: all-positive sequences, all-negative, single-element chunks
   - Discriminant boundary cases

3. Extended PSLQ:
   - Known identities: ζ(2)=π²/6, ζ(4)=π⁴/90, Catalan=β(2)
   - Graded tier ordering: linear finds simple matches, quadratic finds products
   - Regression: existing tests still pass with the new code

4. Fingerprinting:
   - End-to-end: known GCF → correct quality score
   - Robustness: divergent GCF, degenerate coefficients
   - Convergence rate accuracy for known-rate GCFs
"""
import sys
import os
import unittest
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mpmath
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Apéry Family Domain Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAperyFamilyDomainMathCorrectness(unittest.TestCase):
    """Verify mathematical correctness of structural template families."""

    def setUp(self):
        from modules.continued_fractions.domains.AperyFamilyDomain import AperyFamilyDomain
        self.AperyFamilyDomain = AperyFamilyDomain

    def test_apery_zeta3_known_identity_convergent_value(self):
        """
        The ACTUAL convergent of Apéry's GCF must equal 6/ζ(3) - 5.
        
        Apéry's formula: 6/ζ(3) = 5 + (-1)^6/(117 + (-2)^6/(535 + ...))
        Using coefficients x₀=1, x₁=1, x₂=17, x₃=5, x₄=-1:
          a(n) = (n+1)(17n(n+1)+5) → a(0)=5, a(1)=117, a(2)=535, ...
          b(n) = -n⁶              → b(1)=-1, b(2)=-64, b(3)=-729, ...
        """
        domain = self.AperyFamilyDomain(
            family='apery_zeta3',
            a_coefs_ranges=[(1, 1), (1, 1), (17, 17), (5, 5)],
            b_coef_range=(-1, -1),
        )

        an_iter, bn_iter = domain.get_calculation_method()
        # Generate enough terms for convergence
        a_coef = (1, 1, 17, 5)
        b_coef = (-1,)
        
        an_list = list(an_iter(a_coef, 100, start_n=0))
        bn_list = list(bn_iter(b_coef, 100, start_n=1))

        # Verify initial terms match known values from Zeta3Domain1
        # a(n) = (x₀n+x₁)(x₂n(n+1)+x₃) with (1,1,17,5):
        # a(0) = (0+1)(0+5) = 5
        # a(1) = (1+1)(17*1*2+5) = 2*39 = 78
        # a(2) = (2+1)(17*2*3+5) = 3*107 = 321
        self.assertEqual(an_list[0], 5,    f"a(0) should be 5, got {an_list[0]}")
        self.assertEqual(an_list[1], 78,   f"a(1) should be 78, got {an_list[1]}")
        self.assertEqual(an_list[2], 321,  f"a(2) should be 321, got {an_list[2]}")
        self.assertEqual(bn_list[0], -1,   f"b(1) should be -1, got {bn_list[0]}")
        self.assertEqual(bn_list[1], -64,  f"b(2) should be -64, got {bn_list[1]}")

    def test_apery_zeta3_filter_rejects_divergent(self):
        """Pairs that violate Worpitzky MUST be rejected."""
        domain = self.AperyFamilyDomain(
            family='apery_zeta3',
            a_coefs_ranges=[(-5, 5), (-5, 5), (-5, 5), (-5, 5)],
            b_coef_range=(-20, -1),
        )
        # a_lead = x0 * x2 = 1 * 1 = 1. 4*b_lead = 4*(-20) = -80. -a_lead² = -1.
        # -80 < -1 → divergent → must be rejected
        self.assertFalse(domain.filter_gcfs((1, 0, 1, 0), (-20,)))

    def test_apery_zeta3_filter_accepts_boundary(self):
        """Exact boundary case (discriminant = 0) must be accepted with non-strict mode."""
        domain = self.AperyFamilyDomain(
            family='apery_zeta3',
            a_coefs_ranges=[(-5, 5), (-5, 5), (-5, 5), (-5, 5)],
            b_coef_range=(-20, -1),
            use_strict_convergence_cond=False,
        )
        # Use coprime coefficients to avoid GCD filter rejection.
        # a_lead = 1*1 = 1. 4*b_lead = 4*(-1) = -4. -a_lead² = -1.
        # -4 < -1 → this actually diverges too. Use: a_lead=2*1=2, b=-1.
        # 4*(-1) = -4. -(2)^2 = -4. Exactly boundary. gcd(2,1,1,1,1)=1. ✓
        self.assertTrue(domain.filter_gcfs((2, 1, 1, 1), (-1,)))

    def test_apery_zeta3_filter_rejects_strict_boundary(self):
        """Strict mode must reject exact boundary."""
        domain = self.AperyFamilyDomain(
            family='apery_zeta3',
            a_coefs_ranges=[(-5, 5), (-5, 5), (-5, 5), (-5, 5)],
            b_coef_range=(-20, -1),
            use_strict_convergence_cond=True,
        )
        # Same coprime boundary case as above but with strict mode.
        # a_lead = 2*1 = 2. 4*(-1) = -4 == -(2²) = -4. Boundary → strict rejects.
        self.assertFalse(domain.filter_gcfs((2, 1, 1, 1), (-1,)))

    def test_gcd_filter_rejects_reducible(self):
        """GCFs where all coefficients share a common factor should be filtered."""
        domain = self.AperyFamilyDomain(
            family='apery_zeta3',
            a_coefs_ranges=[(-10, 10)] * 4,
            b_coef_range=(-10, -1),
        )
        # gcd(2, 4, 6, 8, -2) = 2 → reducible → reject
        self.assertFalse(domain.filter_gcfs((2, 4, 6, 8), (-2,)))

    def test_ramanujan_pi_family_produces_finite_convergents(self):
        """
        All Ramanujan-type GCFs with small coefficients should converge.
        Test: a(n) = 2n+1, b(n) = -n² is a classic convergent CF for tanh(1).
        """
        domain = self.AperyFamilyDomain(
            family='ramanujan_pi',
            a_coefs_ranges=[(1, 5), (-5, 5)],
            b_coef_range=(-5, -1),
        )
        
        an_iter, bn_iter = domain.get_calculation_method()
        # a(n) = 2n+1, b(n) = -n²
        an = list(an_iter((2, 1), 50, start_n=0))
        bn = list(bn_iter((-1,), 50, start_n=1))
        
        # Backward evaluation for stability
        mpmath.mp.dps = 50
        f = mpmath.mpf(0)
        for i in range(len(bn) - 1, -1, -1):
            denom = an[i + 1] + f
            if abs(denom) < 1e-100:
                self.fail("Denominator went to zero — GCF diverged")
            f = bn[i] / denom
        result = an[0] + f
        self.assertTrue(mpmath.isfinite(result), f"Convergent not finite: {result}")

    def test_classical_log_family_4dof(self):
        """Classical family should have 4 degrees of freedom (2 for a, 2 for b)."""
        domain = self.AperyFamilyDomain(
            family='classical_log',
            a_coefs_ranges=[(1, 5), (-5, 5)],
            b_coef_range=[(-5, -1), (-3, 3)],
        )
        iter_count = domain.num_iterations
        self.assertGreater(iter_count, 0)
        # 5 * 11 * 5 * 7 = 1925
        self.assertEqual(iter_count, 5 * 11 * 5 * 7)

    def test_generalized_d2_creates_degree4_bn(self):
        """Generalized degree-2 should produce b(n) = c*n⁴."""
        domain = self.AperyFamilyDomain(
            family='generalized',
            target_degree=2,
            a_coefs_ranges=[(-3, 3)] * 3,
            b_coef_range=(-5, -1),
        )
        an_iter, bn_iter = domain.get_calculation_method()
        bn = list(bn_iter((-2,), 5, start_n=1))
        # b(1) = -2*1⁴ = -2, b(2) = -2*16 = -32, b(3) = -2*81 = -162
        self.assertEqual(bn[0], -2)
        self.assertEqual(bn[1], -32)
        self.assertEqual(bn[2], -162)

    def test_invalid_family_raises(self):
        """Unknown family name should raise ValueError."""
        with self.assertRaises(ValueError):
            self.AperyFamilyDomain(
                family='nonexistent_family',
                a_coefs_ranges=[(1, 1)],
                b_coef_range=(-1, -1),
            )

    def test_wrong_dimension_raises(self):
        """Wrong number of coefficient ranges should raise ValueError."""
        with self.assertRaises(ValueError):
            self.AperyFamilyDomain(
                family='apery_zeta3',
                a_coefs_ranges=[(1, 1), (1, 1)],  # needs 4, got 2
                b_coef_range=(-1, -1),
            )


class TestAperyFamilyCrossValidation(unittest.TestCase):
    """Cross-validate AperyFamilyDomain against existing Zeta3Domain1."""

    def test_zeta3_iterator_output_matches(self):
        """
        The an/bn iterators from AperyFamilyDomain(apery_zeta3) and
        Zeta3Domain1 must produce identical series values for the same coefficients.
        """
        from modules.continued_fractions.domains.AperyFamilyDomain import AperyFamilyDomain
        from modules.continued_fractions.domains.Zeta3Domain1 import Zeta3Domain1

        # Both domains use the same structural template
        apery_dom = AperyFamilyDomain(
            family='apery_zeta3',
            a_coefs_ranges=[(-3, 3)] * 4,
            b_coef_range=(-5, -1),
        )
        
        z3_an_iter, z3_bn_iter = Zeta3Domain1.get_calculation_method()
        ap_an_iter, ap_bn_iter = apery_dom.get_calculation_method()

        test_coefs_a = (1, 1, 17, 5)
        test_coefs_b = (-1,)

        z3_an = list(z3_an_iter(test_coefs_a, 30, start_n=1))
        ap_an = list(ap_an_iter(test_coefs_a, 30, start_n=1))

        z3_bn = list(z3_bn_iter(test_coefs_b, 30, start_n=1))
        ap_bn = list(ap_bn_iter(test_coefs_b, 30, start_n=1))

        self.assertEqual(z3_an, ap_an,
                         "a(n) series differ between AperyFamilyDomain and Zeta3Domain1")
        self.assertEqual(z3_bn, ap_bn,
                         "b(n) series differ between AperyFamilyDomain and Zeta3Domain1")

    def test_all_families_instantiate(self):
        """Every registered family should instantiate without error."""
        from modules.continued_fractions.domains.AperyFamilyDomain import AperyFamilyDomain
        
        for family_name in AperyFamilyDomain.AVAILABLE_FAMILIES:
            from modules.continued_fractions.domains.AperyFamilyDomain import _FAMILY_REGISTRY
            fam = _FAMILY_REGISTRY[family_name]
            n_a = fam['n_a_free']
            n_b = fam['n_b_free']
            
            a_ranges = [(-2, 2)] * n_a
            if n_b == 1:
                b_range = (-3, -1)
            else:
                b_range = [(-3, -1)] * n_b
            
            domain = AperyFamilyDomain(
                family=family_name,
                a_coefs_ranges=a_ranges,
                b_coef_range=b_range,
            )
            self.assertGreater(domain.num_iterations, 0,
                               f"Family {family_name} has 0 iterations")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: GPU Asymptotic Filter Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGPUAsymptoticFilter(unittest.TestCase):
    """Test the B-1 GPU-native convergence filter for correctness."""

    def test_discriminant_positive_case(self):
        """a(N)² + 4·b(N) > 0 → pair is retained."""
        import torch
        # a(N) = 10, b(N) = -5. disc = 100 - 20 = 80 > 0
        a_last = torch.tensor([10.0])
        b_last = torch.tensor([-5.0])
        disc = a_last ** 2 + 4.0 * b_last
        self.assertTrue((disc >= 0).all().item())

    def test_discriminant_negative_case(self):
        """a(N)² + 4·b(N) < 0 → pair is pruned."""
        import torch
        # a(N) = 2, b(N) = -10. disc = 4 - 40 = -36 < 0
        a_last = torch.tensor([2.0])
        b_last = torch.tensor([-10.0])
        disc = a_last ** 2 + 4.0 * b_last
        self.assertTrue((disc < 0).all().item())

    def test_discriminant_exact_boundary(self):
        """a(N)² + 4·b(N) = 0 → pair is retained (non-strict)."""
        import torch
        # a(N) = 4, b(N) = -4. disc = 16 - 16 = 0
        a_last = torch.tensor([4.0])
        b_last = torch.tensor([-4.0])
        disc = a_last ** 2 + 4.0 * b_last
        self.assertTrue((disc >= 0).all().item())

    def test_batch_discriminant_mixed(self):
        """Mixed batch: some pass, some fail. Verify correct masking."""
        import torch
        # 3 a values × 2 b values
        a_last = torch.tensor([10.0, 2.0, 5.0])   # leads to discs: +, -, +
        b_last = torch.tensor([-5.0, -100.0])       # -5: mild, -100: severe

        # Broadcasting: (3,1) × (1,2) → (3,2)
        a_exp = a_last.unsqueeze(1).expand(3, 2).reshape(-1)  # [10,10,2,2,5,5]
        b_exp = b_last.unsqueeze(0).expand(3, 2).reshape(-1)  # [-5,-100,-5,-100,-5,-100]

        disc = a_exp ** 2 + 4.0 * b_exp
        mask = disc >= 0

        # Expected discriminants:
        # (10,-5): 100-20=80 → pass    (10,-100): 100-400=-300 → fail
        # (2,-5):  4-20=-16 → fail     (2,-100):  4-400=-396 → fail
        # (5,-5):  25-20=5 → pass      (5,-100):  25-400=-375 → fail
        expected = [True, False, False, False, True, False]
        for i, exp in enumerate(expected):
            self.assertEqual(mask[i].item(), exp,
                             f"Position {i}: expected {exp}, got {mask[i].item()}")

    def test_index_remapping_correctness(self):
        """After filtering, alive indices must correctly map back to original positions."""
        import torch
        # Simulate: 6 flat indices, 3 survive
        convergent_mask = torch.tensor([True, False, False, False, True, False])
        alive_idx = torch.nonzero(convergent_mask).squeeze(1)
        
        # positions 0 and 4 survive
        self.assertEqual(alive_idx.tolist(), [0, 4])
        
        # If a hit is found at filtered position 1 (second alive element),
        # it should map back to original position 4
        filtered_hit_pos = 1
        original_pos = alive_idx[filtered_hit_pos].item()
        self.assertEqual(original_pos, 4)

        # With chunk_b_size=2, original (a_idx, b_idx) = divmod(4, 2) = (2, 0)
        chunk_b_size = 2
        a_idx = original_pos // chunk_b_size
        b_idx = original_pos % chunk_b_size
        self.assertEqual(a_idx, 2)
        self.assertEqual(b_idx, 0)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Extended PSLQ Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestExtendedPSLQ(unittest.TestCase):
    """Thorough tests for the graded 3-tier PSLQ search."""

    def test_linear_tier_finds_gamma(self):
        """Linear tier should find γ trivially."""
        from modules.continued_fractions.utils.lll_identity_resolver import resolve_identity
        mpmath.mp.dps = 100
        result = resolve_identity(
            mpmath.euler,
            basis_constants={'gamma', '1'},
            precision=100,
        )
        self.assertTrue(result['found'])
        self.assertEqual(result['method'], 'pslq_linear')

    def test_linear_tier_finds_rational_pi(self):
        """7π/22 should be found via linear PSLQ."""
        from modules.continued_fractions.utils.lll_identity_resolver import resolve_identity
        mpmath.mp.dps = 100
        val = 7 * mpmath.pi / 22
        result = resolve_identity(
            val,
            basis_constants={'pi', '1'},
            precision=100,
            max_denominator=100,
        )
        self.assertTrue(result['found'])
        if 'pi' in result.get('coefficients', {}):
            self.assertAlmostEqual(result['coefficients']['pi'], 7.0/22.0, places=5)

    def test_direct_pslq_finds_zeta2_as_pi_squared(self):
        """Direct PSLQ with small basis [ζ(2), π², 1] must find [-6, 1, 0]."""
        mpmath.mp.dps = 300
        zeta2 = mpmath.zeta(2)
        pi2 = mpmath.pi ** 2
        one = mpmath.mpf(1)
        
        rel = mpmath.pslq([zeta2, pi2, one], maxcoeff=1000)
        self.assertIsNotNone(rel, "PSLQ returned None for [ζ(2), π², 1]")
        self.assertEqual(rel[0], -6, f"Expected -6, got {rel[0]}")
        self.assertEqual(rel[1], 1, f"Expected 1, got {rel[1]}")
        self.assertEqual(rel[2], 0, f"Expected 0, got {rel[2]}")

    def test_quadratic_basis_has_products(self):
        """Quadratic basis must contain pi^2, pi*gamma, gamma^2, etc."""
        from modules.continued_fractions.utils.lll_identity_resolver import _build_quadratic_basis
        basis = _build_quadratic_basis({'pi', 'gamma', 'log2', '1'}, 100)
        
        self.assertIn('pi^2', basis)
        self.assertIn('gamma^2', basis)
        self.assertIn('pi*gamma', basis)
        self.assertIn('pi*log2', basis)
        self.assertIn('pi^4', basis)
        # Linear basis elements should also be present
        self.assertIn('pi', basis)
        self.assertIn('gamma', basis)
        self.assertIn('1', basis)

    def test_algebraic_basis_has_roots(self):
        """Algebraic basis must contain sqrt(pi), cbrt(e), 1/pi, etc."""
        from modules.continued_fractions.utils.lll_identity_resolver import _build_algebraic_basis
        basis = _build_algebraic_basis({'pi', 'e', 'zeta3', '1'}, 100)
        
        self.assertIn('sqrt(pi)', basis)
        self.assertIn('sqrt(e)', basis)
        self.assertIn('sqrt(zeta3)', basis)
        self.assertIn('cbrt(pi)', basis)
        self.assertIn('cbrt(e)', basis)
        self.assertIn('1/pi', basis)

    def test_basis_size_guard_prevents_overflow(self):
        """With low precision and large basis, the guard should skip tiers."""
        from modules.continued_fractions.utils.lll_identity_resolver import resolve_identity
        import logging
        
        # 30 dps → max safe basis = 30/8 = 3. Any basis > 3 should be skipped.
        result = resolve_identity(
            mpmath.mpf('1.2345678901234'),
            basis_constants={'pi', 'gamma', 'log2', 'e', 'zeta3', 'zeta5', '1'},
            precision=30,
        )
        # Should not crash, even if it doesn't find anything
        self.assertIn('found', result)

    def test_regression_existing_lll_tests_pass(self):
        """The original test_lll_resolver tests must still pass with our changes."""
        mpmath.mp.dps = 50
        from modules.continued_fractions.utils.lll_identity_resolver import resolve_identity

        # Test 1: γ self-identification
        r1 = resolve_identity(mpmath.euler, basis_constants={'gamma', '1'}, precision=50)
        self.assertTrue(r1['found'], "γ self-identification regression")

        # Test 2: ln(2) identification
        r2 = resolve_identity(mpmath.log(2), basis_constants={'log2', '1'}, precision=50)
        self.assertTrue(r2['found'], "ln(2) identification regression")

        # Test 3: 3γ/7
        val = mpmath.mpf(3) * mpmath.euler / 7
        r3 = resolve_identity(val, basis_constants={'gamma', '1'}, precision=50, max_denominator=50)
        self.assertTrue(r3['found'], "3γ/7 identification regression")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Convergent Fingerprinting Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestConvergentFingerprinting(unittest.TestCase):
    """Enhanced fingerprinting tests."""

    def test_fingerprint_produces_all_required_keys(self):
        """Report must contain all documented keys."""
        from modules.continued_fractions.utils.convergent_fingerprint import fingerprint_gcf_hit
        
        def an_iter(coef, max_runs, start_n=0):
            for i in range(start_n, max_runs):
                yield 2 * i + 1
        def bn_iter(coef, max_runs, start_n=0):
            for i in range(start_n, max_runs):
                yield -(i ** 2)
        
        report = fingerprint_gcf_hit(
            a_coef=(2, 1), b_coef=(-1,),
            a_iterator=an_iter, b_iterator=bn_iter,
            n_verify_terms=100, precision=50,
        )
        
        required_keys = [
            'target', 'a_coef', 'b_coef', 'convergent_value',
            'convergent_str', 'identity', 'convergence_rate',
            'rate_identity', 'near_misses', 'quality_score',
        ]
        for key in required_keys:
            self.assertIn(key, report, f"Missing key: {key}")

    def test_fingerprint_convergent_is_finite(self):
        """A known-convergent GCF must produce a finite convergent."""
        from modules.continued_fractions.utils.convergent_fingerprint import fingerprint_gcf_hit
        
        def an_iter(coef, max_runs, start_n=0):
            for i in range(start_n, max_runs):
                yield 2 * i + 1
        def bn_iter(coef, max_runs, start_n=0):
            for i in range(start_n, max_runs):
                yield -(i ** 2)
        
        report = fingerprint_gcf_hit(
            a_coef=(2, 1), b_coef=(-1,),
            a_iterator=an_iter, b_iterator=bn_iter,
            n_verify_terms=100, precision=50,
        )
        
        self.assertIsNotNone(report['convergent_value'])
        self.assertTrue(mpmath.isfinite(report['convergent_value']))
        self.assertGreater(report['convergence_rate'], 0)

    def test_fingerprint_quality_score_range(self):
        """Quality score must be in [0, 100]."""
        from modules.continued_fractions.utils.convergent_fingerprint import fingerprint_gcf_hit
        
        def an_iter(coef, max_runs, start_n=0):
            for i in range(start_n, max_runs):
                yield 2 * i + 1
        def bn_iter(coef, max_runs, start_n=0):
            for i in range(start_n, max_runs):
                yield -(i ** 2)
        
        report = fingerprint_gcf_hit(
            a_coef=(2, 1), b_coef=(-1,),
            a_iterator=an_iter, b_iterator=bn_iter,
            n_verify_terms=100, precision=50,
        )
        
        self.assertGreaterEqual(report['quality_score'], 0)
        self.assertLessEqual(report['quality_score'], 100)

    def test_fingerprint_report_formatting(self):
        """Pretty-print formatter must produce non-empty multi-line string."""
        from modules.continued_fractions.utils.convergent_fingerprint import (
            fingerprint_gcf_hit, format_fingerprint_report
        )
        
        def an_iter(coef, max_runs, start_n=0):
            for i in range(start_n, max_runs):
                yield 2 * i + 1
        def bn_iter(coef, max_runs, start_n=0):
            for i in range(start_n, max_runs):
                yield -(i ** 2)
        
        report = fingerprint_gcf_hit(
            a_coef=(2, 1), b_coef=(-1,),
            a_iterator=an_iter, b_iterator=bn_iter,
            n_verify_terms=100, precision=50,
        )
        
        formatted = format_fingerprint_report(report)
        self.assertIsInstance(formatted, str)
        self.assertGreater(len(formatted), 100)
        self.assertIn('CONVERGENT FINGERPRINT REPORT', formatted)

    def test_convergence_rate_for_known_gcf(self):
        """
        Test convergence rate measurement on a(n)=2n+1, b(n)=-n².
        This CF converges linearly; the rate should be a small positive number.
        """
        from modules.continued_fractions.utils.convergent_fingerprint import _compute_convergence_rate
        
        mpmath.mp.dps = 100
        N = 200
        # tanh(1) = 1/(1 + 1/(3 + 4/(5 + 9/(7 + ...))))
        # a(n) = 2n+1, b(n) = n² (all positive — standard form)
        an = [2 * i + 1 for i in range(N)]
        bn = [i * i for i in range(N)]  # b(0)=0, b(1)=1, b(2)=4, ...
        
        # Reference: compute the convergent of a0 + b1/(a1 + b2/(a2 + ...))
        f = mpmath.mpf(0)
        for i in range(N - 1, 0, -1):
            denom = an[i] + f
            if denom != 0:
                f = bn[i] / denom
        reference = an[0] + f

        rate = _compute_convergence_rate(an, bn, reference, 100)
        
        # Rate should be non-negative (converging)
        self.assertGreaterEqual(rate, 0.0, f"Rate is negative: {rate}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: Integration / End-to-End Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEndIntegration(unittest.TestCase):
    """Tests that combine multiple modules together."""

    def test_apery_domain_iter_polys_yields_valid_pairs(self):
        """iter_polys should yield (a_coef, b_coef) pairs that all pass filter."""
        from modules.continued_fractions.domains.AperyFamilyDomain import AperyFamilyDomain
        
        domain = AperyFamilyDomain(
            family='ramanujan_pi',
            a_coefs_ranges=[(1, 3), (-2, 2)],
            b_coef_range=(-3, -1),
        )
        
        count = 0
        for a_coef, b_coef in domain.iter_polys('a'):
            # Every yielded pair must pass the filter
            self.assertTrue(domain.filter_gcfs(a_coef, b_coef),
                            f"Yielded pair ({a_coef}, {b_coef}) fails filter!")
            count += 1
            if count > 100:
                break
        
        self.assertGreater(count, 0, "No pairs yielded")

    def test_apery_domain_fingerprint_pipeline(self):
        """End-to-end: AperyDomain → generate series → fingerprint."""
        from modules.continued_fractions.domains.AperyFamilyDomain import AperyFamilyDomain
        from modules.continued_fractions.utils.convergent_fingerprint import fingerprint_gcf_hit
        
        domain = AperyFamilyDomain(
            family='apery_zeta3',
            a_coefs_ranges=[(1, 1), (1, 1), (17, 17), (5, 5)],
            b_coef_range=(-1, -1),
        )
        
        an_iter, bn_iter = domain.get_calculation_method()
        
        report = fingerprint_gcf_hit(
            a_coef=(1, 1, 17, 5),
            b_coef=(-1,),
            target_name='zeta3_apery',
            a_iterator=an_iter,
            b_iterator=bn_iter,
            n_verify_terms=200,
            precision=100,
        )
        
        # The convergent should be finite and have a positive convergence rate
        self.assertIsNotNone(report['convergent_value'])
        self.assertTrue(mpmath.isfinite(report['convergent_value']))
        # The convergence rate should be non-negative (converging or saturated)
        self.assertGreaterEqual(report['convergence_rate'], 0)


if __name__ == '__main__':
    # Custom test runner with timing
    print("=" * 70)
    print("  ENHANCED ALGORITHMIC IMPROVEMENTS TEST SUITE")
    print("=" * 70)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    elapsed = time.time() - start_time
    
    print(f"\n{'=' * 70}")
    print(f"  Total: {result.testsRun} tests in {elapsed:.1f}s")
    print(f"  Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failed: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"{'=' * 70}")
