import os
import torch
from typing import Tuple, List

from core.interfaces.base_strategy import BoundingStrategy
from modules.continued_fractions.math_ai.models.actor_critic import ActorCriticGCFNetwork
from modules.continued_fractions.math_ai.agents.alpha_tensor_mcts import AlphaTensorMCTS
from modules.continued_fractions.math_ai.environments.EulerMascheroniEnvironment import EulerMascheroniEnvironment
from modules.continued_fractions.math_ai.training.checkpoint import CheckpointManager

class MCTSStrategy(BoundingStrategy):
    """
    Plugin wrapper for the deep reinforcement learning bounds-narrowing agent.
    Inherits the strictly-typed abstract V3 BoundingStrategy interface.
    """
    def __init__(self, pt_filename: str = "em_mcts.pt", env=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = None
        self.env = env if env else EulerMascheroniEnvironment(max_steps=50)
        self._load_network(pt_filename)

    @property
    def strategy_name(self) -> str:
        return "mcts_alpha_tensor"

    def _load_network(self, pt_filename: str):
        # Scan framework for user-loaded weights
        search_paths = [
            os.path.join(os.getcwd(), pt_filename),
            os.path.join(os.getcwd(), 'checkpoints', pt_filename),
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ramanujan_client', 'checkpoints', pt_filename)),
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'checkpoints', pt_filename))
        ]
        
        ckpt_path = None
        for path in search_paths:
            if os.path.exists(path):
                ckpt_path = path
                break
                
        if ckpt_path:
            self.network = ActorCriticGCFNetwork(state_dim=4, hidden_dim=256, action_dim=2).to(self.device)
            try:
                CheckpointManager.load(ckpt_path, self.network, device=self.device)
                self.network.eval()
            except Exception as e:
                print(f"[!] MCTSStrategy failed to load internal PyTorch checkpoint: {e}")
                self.network = None

    def prune_bounds(self, raw_a_bounds: List[List[int]], raw_b_bounds: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Executes internal physics/MCTS rollouts to intelligently restrict continuous Cartesian bounds.
        Fallback to returning bare parameters if Tensor memory is corrupted or weights are missing.
        """
        if self.network is None:
            return raw_a_bounds, raw_b_bounds
            
        mcts = AlphaTensorMCTS(env=self.env, network=self.network, num_simulations=10)
        
        a_refined, b_refined = mcts.get_action_for_bounds(
            initial_state=self.env.reset(),
            original_a_range=raw_a_bounds,
            original_b_range=raw_b_bounds,
            radius_multiplier=14.0
        )
        return a_refined, b_refined
