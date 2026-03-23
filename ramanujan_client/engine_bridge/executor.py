import os
import sys
import torch
from concurrent.futures import ThreadPoolExecutor

try:
    from ramanujan.LHSHashTable import LHSHashTable
except ModuleNotFoundError:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.append(repo_root)
    from ramanujan.LHSHashTable import LHSHashTable

from ramanujan.constants import g_const_dict
from ramanujan.poly_domains.CartesianProductPolyDomain import CartesianProductPolyDomain
from ramanujan.enumerators.GPUEfficientGCFEnumerator import GPUEfficientGCFEnumerator

# RL integration
from ramanujan.math_ai.models.actor_critic import ActorCriticGCFNetwork
from ramanujan.math_ai.agents.alpha_tensor_mcts import AlphaTensorMCTS
from ramanujan.math_ai.environments.EulerMascheroniEnvironment import EulerMascheroniEnvironment
from ramanujan.math_ai.training.checkpoint import CheckpointManager


class RamanujanExecutor:
    """
    Acts as the bridge between the lightweight client dynamic bounds 
    and the heavyweight PyTorch multi-threaded V2 execution engine.
    
    Now integrates Neural MCTS to dynamically narrow Phase Spaces 
    BEFORE passing the bounding box to the GPU exhaust loop.
    """
    def __init__(self):
        local_db = os.path.join(os.path.dirname(__file__), '..', 'euler_mascheroni.db')
        repo_db = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'euler_mascheroni.db'))
        self.lhs_db_path = local_db if os.path.exists(local_db) else (repo_db if os.path.exists(repo_db) else "euler_mascheroni.db")
        self.const_name = "euler-mascheroni"
        self.const_val = g_const_dict[self.const_name]
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine ThreadPool compute capacity for mpmath verifier
        self.cpu_workers = max(1, os.cpu_count() // 2)
        
        # Load pre-trained RL model if available
        self.network = None
        self._load_rl_checkpoint()

    def _load_rl_checkpoint(self):
        """Attempts to load the trained RL checkpoint to enable smart bounds narrowing."""
        # Look in ramanujan_client/checkpoints/ or repo_root/checkpoints/
        local_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints'))
        
        ckpt_path = None
        for path in [os.path.join(local_dir, 'em_mcts.pt'), os.path.join(repo_dir, 'em_mcts.pt')]:
            if os.path.exists(path):
                ckpt_path = path
                break
                
        if ckpt_path:
            print(f"[*] Found RL checkpoint: {ckpt_path}. Neural guidance ENABLED.")
            # Initialize network matching the training script shape
            self.network = ActorCriticGCFNetwork(state_dim=4, hidden_dim=256, action_dim=2).to(self.device)
            try:
                CheckpointManager.load(ckpt_path, self.network, device=self.device)
                self.network.eval()
            except Exception as e:
                print(f"[!] Failed to load checkpoint: {e}. Falling back to brute force.")
                self.network = None
        else:
            print(f"[*] No RL checkpoint found (`checkpoints/em_mcts.pt`). Neural guidance DISABLED (using pure Cartesian exhaust).")

    def _apply_neural_bounds(self, work_unit) -> tuple:
        """Uses Neural MCTS rollouts to intelligently restrict the raw Cartesian bounds."""
        a_range = work_unit.get('a_coef_range')
        b_range = work_unit.get('b_coef_range')
        
        if self.network is None:
            return a_range, b_range
            
        print(f"[*] AI Agent calculating optimal bounds Sub-Space via MCTS rollouts...")
        env = EulerMascheroniEnvironment(max_steps=50) # Just enough steps to gauge trajectory
        
        # 10 rollouts since we just want a quick inference bound (not full tree search)
        mcts = AlphaTensorMCTS(env=env, network=self.network, num_simulations=10)
        
        a_refined, b_refined = mcts.get_action_for_bounds(
            initial_state=env.reset(),
            original_a_range=a_range,
            original_b_range=b_range,
            radius_multiplier=4.0  # Safe bounds
        )
        
        print(f"    Raw structural bounds: a={a_range}, b={b_range}")
        print(f"    AI-refined bounds:     a={a_refined}, b={b_refined}")
        return a_refined, b_refined

    def execute_work_unit(self, work_unit):
        print(f"\n==================================================")
        if "tier" in work_unit: # V1 fallback
            print(f"[*] Engine V2: Work Unit #{work_unit['id']} [{work_unit['tier'].upper()}]")
        else:
            print(f"[*] Engine V2: Work Unit V2-Dynamic-Chunk")
        print(f"[*] Target Constant: {work_unit['constant_name']}")
        print(f"==================================================\n")
        
        a_deg = work_unit['a_deg']
        b_deg = work_unit['b_deg']
        
        # 1. Initialize the LHS Hash Table
        # (Must exist; user should run script if it doesn't)
        if not os.path.exists(self.lhs_db_path):
            raise FileNotFoundError(f"Missing {self.lhs_db_path}. Please run `scripts/seed_euler_mascheroni_db.py` first.")
            
        lhs = LHSHashTable(self.lhs_db_path, 30, [self.const_val])
        
        # 2. Refine bounds using Deep RL MCTS
        a_refined, b_refined = self._apply_neural_bounds(work_unit)
        
        poly_search_domain = CartesianProductPolyDomain(
            a_deg=a_deg, a_coef_range=[0, 0],
            b_deg=b_deg, b_coef_range=[0, 0]
        )
        # Apply the explicit bounds
        poly_search_domain.a_coef_range = a_refined
        poly_search_domain.b_coef_range = b_refined
        poly_search_domain._setup_metadata()
        
        evaluations = poly_search_domain.get_an_length() * poly_search_domain.get_bn_length()
        print(f"[*] Execution Volume: {evaluations:,} mathematically strict combinations to exhaust")
        
        # 3. Spin up the advanced GPUEfficientGCFEnumerator
        enumerator = GPUEfficientGCFEnumerator(
            lhs,
            poly_search_domain,
            [self.const_val]
        )
        
        # Execute the bounds!
        verified_hits = enumerator.full_execution(verbose=True)
        
        print(f"\n[+] Processing chunk finished. Found {len(verified_hits)} ultra-verified hits.")
        return verified_hits
