#!/usr/bin/env python3
"""
Research-Scale MCTS Visual Evaluator
======================================

Loads a trained checkpoint and visually outputs the MCTS tree decision process
step-by-step for the Euler-Mascheroni convergence mapping.

Useful for physically watching the agent narrow the coordinate bounds before
handing off to the GPUEfficientGCFEnumerator in the client.

Usage:
    python research_training/eval_mcts.py
"""

import os
import sys
import yaml
import torch
import numpy as np

# Ensure repo root is available
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ramanujan.math_ai.models.actor_critic import ActorCriticGCFNetwork
from ramanujan.math_ai.agents.alpha_tensor_mcts import AlphaTensorMCTS
from ramanujan.math_ai.environments.EulerMascheroniEnvironment import EulerMascheroniEnvironment
from ramanujan.math_ai.training.checkpoint import CheckpointManager

def load_config(path='research_training/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_evaluation():
    config = load_config()
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    
    ckpt_path = os.path.join(_REPO_ROOT, "checkpoints", config['training']['checkpoint_filename'])
    if not os.path.exists(ckpt_path):
        print(f"[!] No checkpoint found at {ckpt_path}. Please train the model first.")
        return
        
    print(f"\n==================================================")
    print(f"      MCTS Neural Bounding Visualizer (Eval)      ")
    print(f"==================================================\n")
    print(f"[*] Loading Brain... {ckpt_path}")
    
    network = ActorCriticGCFNetwork(
        state_dim=config['network']['state_dim'],
        hidden_dim=config['network']['hidden_dim'],
        action_dim=config['network']['action_dim'],
    ).to(device)
    
    ckpt = CheckpointManager.load(ckpt_path, network, device=device)
    network.eval()
    
    print(f"\n[*] Model Checkpoint Stats:")
    print(f"    - Episode: {ckpt.get('episode', 'Unknown')}")
    print(f"    - Best Eval Reward: {ckpt.get('best_mean_reward', 'Unknown'):.4f}")
    
    env = EulerMascheroniEnvironment(max_steps=10)
    mcts = AlphaTensorMCTS(
        env=env,
        network=network,
        num_simulations=config['mcts']['simulations'],
        c_puct=config['mcts']['c_puct']
    )
    
    initial_state = env.reset()
    
    # Example boundary
    original_a_range = [[-10, 10], [-10, 10], [-10, 10]]
    original_b_range = [[-10, 10], [-10, 10], [-10, 10]]
    
    print("\n-------------------------------------------------------------")
    print("                Simulating MCTS Depth Search                 ")
    print("-------------------------------------------------------------")
    
    action, visit_probs = mcts.search(initial_state)
    a_proxy, b_proxy = float(action[0]), float(action[1])
    
    print(f"\n[💡] MCTS Search Complete! ({config['mcts']['simulations']} Rollouts)")
    print(f"     Prior Confidence Profile: {visit_probs}")
    print(f"     Winning Latent Action: a_proxy={a_proxy:+.3f}, b_proxy={b_proxy:+.3f}")
    
    # Calculate physical restriction bounds
    a_refined, b_refined = mcts.get_action_for_bounds(
        initial_state, original_a_range, original_b_range, radius_multiplier=4.0
    )
    
    print(f"\n[📉] Final Cartesian Bounding Application:")
    print(f"     Original 'a' Bounds: {original_a_range}")
    print(f"     MCTS Restricted 'a': {a_refined}")
    print(f"     ---------------------------------------------")
    print(f"     Original 'b' Bounds: {original_b_range}")
    print(f"     MCTS Restricted 'b': {b_refined}")
    print(f"\n[+] The client GPU enumerator will now execute over these restricted ranges!")

if __name__ == '__main__':
    run_evaluation()
