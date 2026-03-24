import unittest
from modules.continued_fractions.engines.FREnumerator import FREnumerator
from modules.continued_fractions.domains.Zeta5Domain import Zeta5Domain
from modules.continued_fractions.targets import g_const_dict

class TestFRExpansion(unittest.TestCase):
    def test_multidimensional_pslq_setup(self):
        poly_search_domain = Zeta5Domain(
            [(1, 1), (0, 0), (0, 0)],
            (1, 1))

        # Two constants (Zeta3, Zeta5) -> Multi-dimensional space should be:
        # [1, zeta3, zeta5, zeta3*zeta3, zeta3*zeta5, zeta5*zeta5] -> Length 6
        enumerator = FREnumerator(
            poly_search_domain,
            [g_const_dict['zeta'](3), g_const_dict['zeta'](5)]
        )
        
        # Test number of dimensions constructed
        consts = [gen() for gen in enumerator.constants_generator]
        numer_items = [1] + consts
        for i in range(len(consts)):
            for j in range(i, len(consts)):
                numer_items.append(consts[i] * consts[j])
                
        self.assertEqual(len(numer_items), 6)

if __name__ == '__main__':
    unittest.main()
