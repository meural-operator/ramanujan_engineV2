import unittest
import math
from modules.continued_fractions.math_ai.agents.program_sandbox import (
    is_safe,
    compile_lambda,
    evaluate_sequence,
    evaluate_gcf_fitness
)

class TestProgramSandbox(unittest.TestCase):
    
    def test_is_safe_whitelisted(self):
        self.assertTrue(is_safe("lambda n: n**2 + 2*n + 1"))
        self.assertTrue(is_safe("lambda n: (-1)**n * n"))
        self.assertTrue(is_safe("lambda n: max(n, 1)"))
        
    def test_is_safe_blacklisted(self):
        self.assertFalse(is_safe("lambda n: __import__('os').system('echo hack')"))
        self.assertFalse(is_safe("lambda n: eval('1+1')"))
        self.assertFalse(is_safe("def f(n): return n")) # No lambda
        
    def test_compile_lambda_success(self):
        func = compile_lambda("lambda n: 2*n + 1")
        self.assertIsNotNone(func)
        self.assertEqual(func(5), 11)
        
    def test_evaluate_sequence(self):
        func = compile_lambda("lambda n: 2*n")
        seq = evaluate_sequence(func, 5)
        self.assertIsNotNone(seq)
        self.assertEqual(seq, [0.0, 2.0, 4.0, 6.0, 8.0])
        
    def test_evaluate_sequence_overflow_protection(self):
        # Extremely large growth should be caught
        func = compile_lambda("lambda n: 1000**n")
        seq = evaluate_sequence(func, 50)
        # Should return None when hitting overflow threshold
        self.assertIsNone(seq)
        
    def test_evaluate_gcf_fitness_golden_ratio(self):
        # Golden ratio CF is simply [1; 1, 1, 1, ...]
        # Here a(0) = 1, b(n) = 1, a(n) = 1
        a_n_code = "lambda n: 1"
        b_n_code = "lambda n: 1"
        
        target = (1 + math.sqrt(5)) / 2
        result = evaluate_gcf_fitness(a_n_code, b_n_code, target, n_terms=50)
        
        self.assertTrue(result['valid'])
        # 50 terms of golden ratio gets ~10 digits of accuracy
        self.assertGreater(result['fitness'], 9.0)
        self.assertGreater(result['convergence_rate'], 0.0)

    def test_evaluate_gcf_fitness_pi_brouncker(self):
        # Brouncker's generalized continued fraction for 4/pi
        # target is 4/pi = 1 + 1^2 / (2 + 3^2 / (2 + ...))
        # a_0 = 1, a_n = 2
        # b_0 = N/A, b_n = (2n-1)^2
        a_n_code = "lambda n: 1 if n == 0 else 2"
        b_n_code = "lambda n: (2*n - 1)**2"
        
        # 4 / pi = a0 + b1/(a1 + b2/(a2 + ...))
        target = 4 / math.pi
        
        result = evaluate_gcf_fitness(a_n_code, b_n_code, target, n_terms=100)
        self.assertTrue(result['valid'])
        self.assertGreater(result['fitness'], 1.0) # Very slow convergence


if __name__ == '__main__':
    unittest.main()
