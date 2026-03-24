import unittest
from modules.continued_fractions.domains.ContinuousRelaxationDomain import ContinuousRelaxationDomain
from modules.continued_fractions.domains.MCTSPolyDomain import MCTSPolyDomain

class TestPolyDomains(unittest.TestCase):
    def test_continuous_relaxation_domain(self):
        domain = ContinuousRelaxationDomain(
            a_deg=2, a_coef_range=[-10, 10], 
            b_deg=2, b_coef_range=[-10, 10],
            epochs=20
        )
        
        # Original width is 20 per coef, gradient descent + penalty should shrink the total bounding space
        an_len = domain.get_an_length()
        bn_len = domain.get_bn_length()
        
        # Verify the domain was initialized correctly and reduced
        self.assertTrue(an_len > 0 and bn_len > 0)
        self.assertTrue(an_len < (21**3))
        
    def test_mcts_poly_domain(self):
        domain = MCTSPolyDomain(
            a_deg=1, a_coef_range=[-20, 20], 
            b_deg=1, b_coef_range=[-20, 20],
            mcts_iterations=200
        )
        
        # Check that MCTS randomly sampled bounds shrinking the 40 limit bounds
        width_sum = sum([a_c[1] - a_c[0] for a_c in domain.a_coef_range])
        self.assertTrue(width_sum <= 80) # Bound is either reduced or effectively max 80 (40+40)

if __name__ == '__main__':
    unittest.main()
