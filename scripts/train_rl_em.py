#!/usr/bin/env python3
"""
RL Training Script: PPO + MCTS for Euler-Mascheroni GCF Discovery
==================================================================

Trains the ActorCriticGCFNetwork using Proximal Policy Optimization (PPO)
in the EulerMascheroniEnvironment. The trained checkpoint is used by the
distributed client nodes to propose optimal coefficient search bounds before
the GPU brute-force enumeration phase.

Usage:
    cd c:\\Users\\DIAT\\ashish\\ramanujan_at_home\\ramanujan_engineV2
    python scripts/train_rl_em.py

    # Full training run (recommended — ~2-4 hours on RTX 4000 Ada):
    python scripts/train_rl_em.py --episodes 100000 --simulations 100

    # Quick smoke test (a few minutes):
    python scripts/train_rl_em.py --episodes 500 --simulations 20 --checkpoint checkpoints/em_mcts_test.pt

The script saves:
    checkpoints/em_mcts.pt          ← Best model (by mean reward over last 100 eps)
    checkpoints/em_mcts_latest.pt   ← Periodic heartbeat checkpoint
    checkpoints/training_state.json ← Metadata (episode, best reward, etc.)
"""

import os
import sys
import argparse
import numpy as np
import torch
from collections import deque
from tqdm import tqdm

# Ensure repo root is on the path when run from scripts/ directory
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from modules.continued_fractions.math_ai.models.actor_critic import ActorCriticGCFNetwork
from modules.continued_fractions.math_ai.environments.EulerMascheroniEnvironment import EulerMascheroniEnvironment
from modules.continued_fractions.math_ai.training.replay_buffer import TrajectoryBuffer
from modules.continued_fractions.math_ai.training.ppo_trainer import PPOTrainer
from modules.continued_fractions.math_ai.training.checkpoint import CheckpointManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO agent for Euler-Mascheroni GCF discovery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--episodes', type=int, default=50000,
                        help='Total training episodes')
    parser.add_argument('--max-steps', type=int, default=100,
                        help='Max GCF recurrence steps per episode')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension of the actor-critic network')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Initial learning rate (Adam)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for GAE')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='GAE lambda (bias-variance trade-off)')
    parser.add_argument('--clip-epsilon', type=float, default=0.2,
                        help='PPO clip epsilon')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='Entropy bonus coefficient (encourages exploration)')
    parser.add_argument('--n-epochs', type=int, default=4,
                        help='PPO epochs per update')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Mini-batch size for PPO updates')
    parser.add_argument('--update-freq', type=int, default=20,
                        help='Number of episodes between PPO updates')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/em_mcts.pt',
                        help='Path to save the best model checkpoint')
    parser.add_argument('--heartbeat', type=int, default=500,
                        help='Episodes between periodic heartbeat saves')
    parser.add_argument('--log-freq', type=int, default=100,
                        help='Episodes between tqdm status updates')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing checkpoint if available')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Device ──────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n[Train] GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    else:
        print(f"\n[Train] ⚠ No GPU detected — training on CPU (will be slow)")
    print(f"[Train] Episodes: {args.episodes:,} | Hidden: {args.hidden_dim} | LR: {args.lr}")
    print(f"[Train] Checkpoint: {args.checkpoint}\n")

    # ── Components ───────────────────────────────────────────────────────────────
    env = EulerMascheroniEnvironment(max_steps=args.max_steps)

    network = ActorCriticGCFNetwork(
        state_dim=4,
        hidden_dim=args.hidden_dim,
        action_dim=2,
    ).to(device)

    trainer = PPOTrainer(
        network=network,
        device=device,
        lr=args.lr,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        n_epochs=args.n_epochs,
        mini_batch_size=args.batch_size,
        total_steps=args.episodes,
    )

    buffer = TrajectoryBuffer(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        device=device,
    )

    ckpt_dir = os.path.dirname(args.checkpoint) or 'checkpoints'
    ckpt_filename = os.path.basename(args.checkpoint)
    ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_dir)
    latest_filename = ckpt_filename.replace('.pt', '_latest.pt')

    # ── Resume if requested ───────────────────────────────────────────────────────
    start_episode = 0
    if args.resume and ckpt_mgr.exists(ckpt_filename):
        ckpt = CheckpointManager.load(
            ckpt_mgr.latest_path(ckpt_filename), network, trainer.optimizer, device
        )
        start_episode = ckpt.get('episode', 0)
        print(f"[Train] Resuming from episode {start_episode:,}\n")

    # ── Tracking ──────────────────────────────────────────────────────────────────
    recent_rewards = deque(maxlen=100)  # sliding window mean
    best_mean_reward = ckpt_mgr.best_mean_reward
    total_episodes = args.episodes - start_episode

    pbar = tqdm(
        range(start_episode, args.episodes),
        desc="Training PPO",
        unit="ep",
        dynamic_ncols=True,
        colour='cyan',
    )

    # ── Training Loop ─────────────────────────────────────────────────────────────
    for episode in pbar:
        state = env.reset()
        episode_reward = 0.0
        done = False
        step = 0

        while not done:
            # ── Collect one step ──────────────────────────────────────────────────
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, std, value = network(state_t)

            dist = torch.distributions.Normal(mean, std)
            action_t = dist.sample()
            log_prob = dist.log_prob(action_t).sum(dim=-1)

            action = action_t.squeeze(0).cpu().numpy()
            value_s = value.squeeze().item()
            log_prob_s = log_prob.item()

            # ── Environment step ──────────────────────────────────────────────────
            next_state, reward, done, info = env.step(action)

            buffer.store(
                state=state,
                action=action,
                reward=reward,
                value=value_s,
                log_prob=log_prob_s,
                done=done,
            )

            state = next_state
            episode_reward += reward
            step += 1

        recent_rewards.append(episode_reward)

        # ── PPO Update ────────────────────────────────────────────────────────────
        if (episode + 1) % args.update_freq == 0 and len(buffer) > 0:
            # Bootstrap value for the terminal state (0 since episode ended)
            last_val_t = torch.tensor([state], dtype=torch.float32, device=device)
            with torch.no_grad():
                _, _, last_val = network(last_val_t)
            last_val_scalar = last_val.item() if not done else 0.0

            buffer.compute_gae(last_value=last_val_scalar)
            metrics = trainer.update(buffer)
            buffer.clear()

        # ── Checkpointing & Logging ───────────────────────────────────────────────
        if len(recent_rewards) >= 10:
            mean_reward = float(np.mean(recent_rewards))

            # Save best model
            ckpt_mgr.save(
                network, trainer.optimizer, episode, mean_reward,
                filename=ckpt_filename,
                extra_meta={"entropy": metrics.get("entropy", 0) if 'metrics' in dir() else 0}
            )

        # Periodic heartbeat checkpoint (always saves, regardless of reward)
        if (episode + 1) % args.heartbeat == 0:
            ckpt_mgr.save_always(network, trainer.optimizer, episode,
                                  float(np.mean(recent_rewards)) if recent_rewards else 0,
                                  filename=latest_filename)

        # tqdm progress bar
        if (episode + 1) % args.log_freq == 0 and recent_rewards:
            mean_r = float(np.mean(recent_rewards))
            pbar.set_postfix({
                "ep_r": f"{episode_reward:.1f}",
                "mean100": f"{mean_r:.2f}",
                "best": f"{ckpt_mgr.best_mean_reward:.2f}",
                "lr": f"{trainer.optimizer.param_groups[0]['lr']:.2e}",
            })

    # ── Final save ────────────────────────────────────────────────────────────────
    pbar.close()
    if recent_rewards:
        final_mean = float(np.mean(recent_rewards))
        ckpt_mgr.save_always(network, trainer.optimizer, args.episodes, final_mean,
                              filename=latest_filename)
        print(f"\n[Train] ✅ Training complete!")
        print(f"[Train] Final mean reward (last 100 eps): {final_mean:.4f}")
        print(f"[Train] Best reward achieved:             {ckpt_mgr.best_mean_reward:.4f}")
        print(f"[Train] Best checkpoint: {ckpt_mgr.latest_path(ckpt_filename)}")
        print(f"\n[Train] To use this checkpoint in the distributed client, copy it to:")
        print(f"        ramanujan_client/checkpoints/em_mcts.pt")


if __name__ == '__main__':
    main()
