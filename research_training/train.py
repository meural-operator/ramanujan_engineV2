#!/usr/bin/env python3
"""
Research-Scale TensoBoard PPO Trainer for Euler-Mascheroni Constant.

Features:
  - Loads hyperparameters from config.yaml
  - Integrates Curriculum Learning (starts easy, graduates to harder GCF depths)
  - Logs comprehensive metrics to TensorBoard (Loss, Entropy, KL, Value, Curriculum Depth)
  - Manages Checkpoints (Dynamic best-model saving + periodic heartbeats)
  - Decaying Exploration (Entropy dynamically lowers as model masters early curriculum levels)

Usage:
    cd c:\\Users\\DIAT\\ashish\\ramanujan_at_home\\ramanujan_engineV2
    python research_training/train.py
"""

import os
import sys
import yaml
import torch
import numpy as np
from collections import deque
from tqdm import tqdm
from datetime import datetime

# Initialize TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Ensure repo root is available
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from modules.continued_fractions.math_ai.models.actor_critic import ActorCriticGCFNetwork
from modules.continued_fractions.math_ai.training.replay_buffer import TrajectoryBuffer
from modules.continued_fractions.math_ai.training.ppo_trainer import PPOTrainer
from modules.continued_fractions.math_ai.training.checkpoint import CheckpointManager
from research_training.env_curriculum import CurriculumEulerMascheroniEnv

def load_config(path='research_training/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_training():
    config = load_config()
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    
    # ── TensorBoard ─────────────────────────────────────────────────────────────
    # Create unique run string based on time
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(os.path.dirname(__file__), 'runs', f'euler_mascheroni_{run_id}')
    writer = SummaryWriter(log_dir=log_dir)
    print(f"\n[📊 TensorBoard] Logging active. To view live training metrics, run:")
    print(f"    tensorboard --logdir research_training/runs/\n")

    # ── Components ───────────────────────────────────────────────────────────────
    env = CurriculumEulerMascheroniEnv(config)
    
    network = ActorCriticGCFNetwork(
        state_dim=config['network']['state_dim'],
        hidden_dim=config['network']['hidden_dim'],
        action_dim=config['network']['action_dim'],
    ).to(device)

    trainer = PPOTrainer(
        network=network,
        device=device,
        lr=config['ppo']['lr'],
        clip_epsilon=config['ppo']['clip_epsilon'],
        entropy_coef=config['ppo']['entropy_coef_initial'],
        value_coef=config['ppo']['value_coef'],
        n_epochs=config['ppo']['n_epochs'],
        max_grad_norm=config['ppo']['max_grad_norm'],
        mini_batch_size=config['ppo']['batch_size'],
        total_steps=config['training']['episodes'],
    )

    buffer = TrajectoryBuffer(
        gamma=config['ppo']['gamma'],
        gae_lambda=config['ppo']['gae_lambda'],
        device=device,
    )

    ckpt_dir = os.path.join(_REPO_ROOT, config['training']['checkpoint_dir'])
    ckpt_filename = config['training']['checkpoint_filename']
    ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_dir)
    latest_filename = ckpt_filename.replace('.pt', '_latest.pt')

    # Resume capability check
    start_episode = 0
    if ckpt_mgr.exists(ckpt_filename):
        print(f"[*] Resuming from existing checkpoint: {ckpt_filename}")
        ckpt = CheckpointManager.load(ckpt_mgr.latest_path(ckpt_filename), network, trainer.optimizer, device)
        start_episode = ckpt.get('episode', 0)

    # ── Tracking ──────────────────────────────────────────────────────────────────
    recent_rewards = deque(maxlen=config['training']['log_freq_episodes'])
    episodes_target = config['training']['episodes']
    
    log_freq = config['training']['log_freq_episodes']
    update_freq = config['ppo']['update_freq']
    heartbeat_freq = config['training']['checkpoint_freq_episodes']
    
    # Entropy Linear Decay variables
    ent_initial = config['ppo']['entropy_coef_initial']
    ent_final = config['ppo']['entropy_coef_final']
    ent_decay_steps = config['ppo']['entropy_decay_steps']
    
    pbar = tqdm(
        range(start_episode, episodes_target),
        desc="[🌌 Training Gamma]",
        unit="ep",
        dynamic_ncols=True,
    )

    # ── Training Loop ─────────────────────────────────────────────────────────────
    for episode in pbar:
        state = env.reset()
        episode_reward = 0.0
        done = False
        
        # ── Dynamic Entropy Decay ────────────────────────────────────────────────
        decay_progress = min(1.0, float(episode) / ent_decay_steps)
        current_entropy_coef = ent_initial - decay_progress * (ent_initial - ent_final)
        trainer.entropy_coef = current_entropy_coef

        # ── Episode Rollout ──────────────────────────────────────────────────────
        while not done:
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, std, value = network(state_t)

            dist = torch.distributions.Normal(mean, std)
            action_t = dist.sample()
            log_prob = dist.log_prob(action_t).sum(dim=-1)

            action = action_t.squeeze(0).cpu().numpy()
            value_s = value.squeeze().item()
            log_prob_s = log_prob.item()

            next_state, reward, done, info = env.step(action)

            buffer.store(
                state=state, action=action, reward=reward,
                value=value_s, log_prob=log_prob_s, done=done,
            )

            state = next_state
            episode_reward += reward

        recent_rewards.append(episode_reward)

        # ── PPO Trajectory Update ────────────────────────────────────────────────
        if (episode + 1) % update_freq == 0 and len(buffer) > 0:
            last_val_t = torch.tensor([state], dtype=torch.float32, device=device)
            with torch.no_grad():
                _, _, last_val = network(last_val_t)
            last_val_scalar = last_val.item() if not done else 0.0

            buffer.compute_gae(last_value=last_val_scalar)
            metrics = trainer.update(buffer)
            buffer.clear()
            
            # Write ML Metrics to Tensorboard
            writer.add_scalar('Loss/Policy', metrics['policy_loss'], episode)
            writer.add_scalar('Loss/Value', metrics['value_loss'], episode)
            writer.add_scalar('Diagnostics/Entropy', metrics['entropy'], episode)
            writer.add_scalar('Diagnostics/KL_Divergence', metrics['approx_kl'], episode)
            writer.add_scalar('Diagnostics/Clip_Fraction', metrics['clip_fraction'], episode)

        # ── Curriculum Evaluation ────────────────────────────────────────────────
        # Ask the wrapper if we should increment GCF depth
        if recent_rewards:
            mean_reward = float(np.mean(recent_rewards))
            if env.check_promotion(mean_reward):
                writer.add_scalar('Curriculum/Max_Depth', env.current_max_steps, episode)

        # ── Checkpointing & Logging ───────────────────────────────────────────────
        if (episode + 1) % log_freq == 0 and recent_rewards:
            mean_r = float(np.mean(recent_rewards))
            
            # Save Best-Model dynamically if mean crossed historical apex
            ckpt_mgr.save(network, trainer.optimizer, episode, mean_r, filename=ckpt_filename)
            
            # Write Reward Metrics to Tensorboard
            writer.add_scalar('Reward/Mean_100_Episodes', mean_r, episode)
            writer.add_scalar('Hyperparameters/Learning_Rate', trainer.optimizer.param_groups[0]['lr'], episode)
            writer.add_scalar('Hyperparameters/Entropy_Coef', current_entropy_coef, episode)

            pbar.set_postfix({
                "Mean_R": f"{mean_r:.1f}",
                "Best_R": f"{ckpt_mgr.best_mean_reward:.1f}",
                "Depth": env.current_max_steps,
                "Ent_C": f"{current_entropy_coef:.3f}",
            })

        # Hearbeat Checkpoint
        if (episode + 1) % heartbeat_freq == 0:
            ckpt_mgr.save_always(network, trainer.optimizer, episode,
                                  float(np.mean(recent_rewards)) if recent_rewards else 0,
                                  filename=latest_filename)

    # ── Shutdown ────────────────────────────────────────────────────────────────
    pbar.close()
    writer.close()
    if recent_rewards:
        final_mean = float(np.mean(recent_rewards))
        ckpt_mgr.save_always(network, trainer.optimizer, episodes_target, final_mean, filename=latest_filename)
        print(f"\n[🏁 Training Concluded]")
        print(f"Final mean reward: {final_mean:.4f}")
        print(f"Absolute best checkpoint saved to: {ckpt_mgr.latest_path(ckpt_filename)}")

if __name__ == '__main__':
    run_training()
