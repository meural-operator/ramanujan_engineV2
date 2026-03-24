import unittest
from modules.continued_fractions.utils.asymptotic_filter import is_asymptotically_convergent

class TestAsymptoticFilter(unittest.TestCase):
    def test_divergent_degrees(self):
        # a_deg * 2 < b_deg -> divergent
        self.assertFalse(is_asymptotically_convergent(a_deg=2, a_leading_coef=1, b_deg=5, b_leading_coef=1))
        
    def test_convergent_degrees(self):
        # a_deg * 2 > b_deg -> convergent
        self.assertTrue(is_asymptotically_convergent(a_deg=3, a_leading_coef=1, b_deg=5, b_leading_coef=1))
        
    def test_worpitzky_boundary(self):
        # a_deg * 2 == b_deg
        # 4 * b_n < -a_n^2 -> diverges (4*-2=-8 < -4)
        self.assertFalse(is_asymptotically_convergent(a_deg=2, a_leading_coef=2, b_deg=4, b_leading_coef=-2))
        
        # 4 * b_n >= -a_n^2 -> converges at boundary (4*-1=-4, -2^2=-4, NOT strictly less)
        self.assertTrue(is_asymptotically_convergent(a_deg=2, a_leading_coef=2, b_deg=4, b_leading_coef=-1))
        
        # Exact boundary - strict vs non-strict
        self.assertTrue(is_asymptotically_convergent(a_deg=2, a_leading_coef=2, b_deg=4, b_leading_coef=-1, strict=False))  # 4*-1 = -4. -2^2 = -4. So >= is met. Wait, 4*b >= -a^2.
        # Actually in filter: if 4 * b_leading_coef < -1 * (a_leading_coef ** 2): return False
        
    def test_worpitzky_logic(self):
        # 4 * (-2) = -8 < -1 * (2**2) = -4 --> False
        self.assertFalse(is_asymptotically_convergent(a_deg=2, a_leading_coef=2, b_deg=4, b_leading_coef=-2))
        
        # 4 * (0) = 0 > -4 --> True
        self.assertTrue(is_asymptotically_convergent(a_deg=2, a_leading_coef=2, b_deg=4, b_leading_coef=0))
        
        # strict equality
        # 4 * (-1) = -4  == -1 * (2**2) = -4
        self.assertTrue(is_asymptotically_convergent(a_deg=2, a_leading_coef=2, b_deg=4, b_leading_coef=-1, strict=False))
        self.assertFalse(is_asymptotically_convergent(a_deg=2, a_leading_coef=2, b_deg=4, b_leading_coef=-1, strict=True))

if __name__ == '__main__':
    unittest.main()
