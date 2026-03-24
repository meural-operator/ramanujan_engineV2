import unittest
from modules.continued_fractions.LHSHashTable import LHSHashTable
from modules.continued_fractions.engines.EfficientGCFEnumerator import EfficientGCFEnumerator
from modules.continued_fractions.engines.GPUEfficientGCFEnumerator import GPUEfficientGCFEnumerator
from modules.continued_fractions.domains.Zeta3Domain1 import Zeta3Domain1
from modules.continued_fractions.targets import g_const_dict
from tests.conjectures_tests import get_testable_data

class TestGPUEfficientGCFEnumerator(unittest.TestCase):
    def test_gpu_matches_cpu(self):
        lhs = LHSHashTable('zeta3.lhs.dept14.db', 14, [g_const_dict['zeta'](3)])

        poly_search_domain = Zeta3Domain1(
            [(2, 2), (1, 1), (1, 17), (1, 5)],  # an coefficients
            (-16, -1)  # bn coefficients
        )

        cpu_enumerator = EfficientGCFEnumerator(
            lhs, poly_search_domain, [g_const_dict['zeta'](3)]
        )
        gpu_enumerator = GPUEfficientGCFEnumerator(
            lhs, poly_search_domain, [g_const_dict['zeta'](3)]
        )

        cpu_results = get_testable_data(cpu_enumerator.full_execution())
        gpu_results = get_testable_data(gpu_enumerator.full_execution())

        # Sets to ignore order
        self.assertEqual(set(cpu_results), set(gpu_results))

class TestGPUEfficientEdgeCases(unittest.TestCase):
    def test_zero_division_parsing(self):
        lhs = LHSHashTable('zeta3.lhs.dept14.db', 14, [g_const_dict['zeta'](3)])

        # A domain engineered to produce zero-division at first step
        poly_search_domain = Zeta3Domain1(
            [(0, 0), (0, 0), (0, 0), (0, 0)], 
            (0, 0) # all zeros
        )

        gpu_enumerator = GPUEfficientGCFEnumerator(
            lhs, poly_search_domain, [g_const_dict['zeta'](3)]
        )

        results = get_testable_data(gpu_enumerator.full_execution())
        # Should gracefully return no matches because a filter weeds out 0-sequences
        self.assertEqual(len(results), 0)

if __name__ == '__main__':
    unittest.main()
