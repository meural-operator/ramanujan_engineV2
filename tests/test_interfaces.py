import unittest
from core.interfaces.base_problem import TargetProblem
from core.interfaces.base_strategy import BoundingStrategy
from core.interfaces.base_engine import ExecutionEngine
from core.interfaces.base_coordinator import NetworkCoordinator

class TestAbstractInterfaces(unittest.TestCase):
    def test_target_constant_abc_enforcement(self):
        class DummyConstant(TargetProblem):
            pass
        with self.assertRaises(TypeError):
            _ = DummyConstant()

    def test_target_constant_valid_implementation(self):
        class ValidConstant(TargetProblem):
            @property
            def name(self): return "tester"
            @property
            def precision(self): return 500
            def generate_lhs_hash_table(self, depth): return {"hash": 0}
            def verify_match(self, a, b): return 1e-150
            
        instance = ValidConstant()
        self.assertEqual(instance.name, "tester")
        self.assertEqual(instance.precision, 500)
        self.assertTrue(instance.verify_match(tuple(), tuple()) < 1e-10)

    def test_bounding_strategy_abc_enforcement(self):
        class DummyStrategy(BoundingStrategy):
            pass
        with self.assertRaises(TypeError):
            _ = DummyStrategy()

    def test_bounding_strategy_valid_implementation(self):
        class ValidStrategy(BoundingStrategy):
            @property
            def strategy_name(self): return "mock_pruner"
            def prune_bounds(self, a, b): return (a, b)
            
        instance = ValidStrategy()
        self.assertEqual(instance.strategy_name, "mock_pruner")
        self.assertEqual(instance.prune_bounds([[1]], [[2]]), ([[1]], [[2]]))

    def test_enumerator_engine_abc_enforcement(self):
        class DummyEngine(ExecutionEngine):
            pass
        with self.assertRaises(TypeError):
            _ = DummyEngine()

    def test_network_coordinator_abc_enforcement(self):
        class DummyCoordinator(NetworkCoordinator):
            pass
        with self.assertRaises(TypeError):
            _ = DummyCoordinator()

if __name__ == "__main__":
    unittest.main()
