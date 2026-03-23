import unittest
import numpy as np
import torch
from ramanujan.math_ai.environments.EulerMascheroniEnvironment import EulerMascheroniEnvironment
from ramanujan.math_ai.models.actor_critic import ActorCriticGCFNetwork
from ramanujan.math_ai.agents.alpha_tensor_mcts import AlphaTensorMCTS
from ramanujan.math_ai.training.replay_buffer import TrajectoryBuffer

class TestMathAIModules(unittest.TestCase):
    
    def test_euler_mascheroni_env(self):
        env = EulerMascheroniEnvironment(max_steps=10)
        state = env.reset()
        
        self.assertEqual(len(state), 4)
        
        # Test step with integer "proxy" actions
        next_state, reward, done, info = env.step(np.array([1.0, 1.0]))
        self.assertEqual(len(next_state), 4)
        self.assertFalse(done)
        self.assertIn("digits_accurate", info)

        # Test overflow guard
        env.q = 1e16
        _, reward, done, info = env.step(np.array([1.0, 1.0]))
        self.assertTrue(done)
        self.assertEqual(reward, -20.0)

    def test_actor_critic_network(self):
        model = ActorCriticGCFNetwork(state_dim=4, hidden_dim=64, action_dim=2)
        state = torch.randn(3, 4) # batch=3, state_dim=4
        
        mean, std, value = model(state)
        self.assertEqual(mean.shape, (3, 2))
        self.assertEqual(std.shape, (3, 2))
        self.assertEqual(value.shape, (3, 1))

        # Test log_prob evaluation
        actions = torch.randn(3, 2)
        log_prob, entropy, val2 = model.evaluate_actions(state, actions)
        self.assertEqual(log_prob.shape, (3,))
        self.assertEqual(entropy.shape, (3,))
        
    def test_replay_buffer(self):
        buf = TrajectoryBuffer()
        
        # Store dummy steps
        for i in range(5):
            buf.store(
                state=np.array([0,0,0,0]), action=np.array([1,1]),
                reward=1.0, value=0.5, log_prob=-1.0, done=(i==4)
            )
            
        self.assertEqual(len(buf), 5)
        buf.compute_gae(last_value=0.0)
        
        self.assertIsNotNone(buf.advantages)
        self.assertIsNotNone(buf.returns)
        
        batches = buf.get_batches(batch_size=2)
        self.assertTrue(len(batches) > 0)

if __name__ == '__main__':
    unittest.main()
