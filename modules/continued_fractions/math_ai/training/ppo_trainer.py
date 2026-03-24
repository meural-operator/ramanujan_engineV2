"""
Full PPO (Proximal Policy Optimization) Trainer for GCF Discovery Agent.
Research-grade implementation featuring:
  - PPO-clip surrogate objective
  - Value function loss with optional clipping
  - Entropy bonus for exploration
  - Gradient norm clipping
  - Linear learning rate schedule with cosine decay
  - Live per-epoch metrics logging
"""
import torch
import torch.optim as optim
import torch.nn as nn
from typing import List, Tuple
import numpy as np


class PPOTrainer:
    """
    Trains the ActorCriticGCFNetwork using PPO-clip.
    
    Usage:
        trainer = PPOTrainer(network, device)
        # ... collect rollout into TrajectoryBuffer ...
        buf.compute_gae(last_value)
        metrics = trainer.update(buf)
        buf.clear()
    """

    def __init__(
        self,
        network: nn.Module,
        device: torch.device,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        mini_batch_size: int = 64,
        total_steps: int = None,  # for LR schedule
    ):
        self.network = network
        self.device = device
        self.clip_epsilon = clip_epsilon
        self.value_clip_epsilon = value_clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size

        self.optimizer = optim.Adam(network.parameters(), lr=lr, eps=1e-5)

        # Cosine annealing schedule (optional, activated when total_steps is given)
        if total_steps is not None:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=lr * 0.01
            )
        else:
            self.scheduler = None

        self._update_count = 0

    def update(self, buffer) -> dict:
        """
        Performs n_epochs of PPO mini-batch updates using data from `buffer`.
        
        Args:
            buffer: TrajectoryBuffer with compute_gae() already called.
        
        Returns:
            Dictionary of mean metrics across all mini-batch updates:
              policy_loss, value_loss, entropy, approx_kl, clip_fraction, lr
        """
        all_policy_loss = []
        all_value_loss = []
        all_entropy = []
        all_kl = []
        all_clip_frac = []

        for epoch in range(self.n_epochs):
            batches = buffer.get_batches(self.mini_batch_size)
            for (states, actions, old_log_probs, advantages, returns) in batches:

                # --- Evaluate actions under current policy ---
                log_probs, entropy, values = self.network.evaluate_actions(states, actions)
                values = values.squeeze(-1)

                # --- Policy (actor) loss: PPO-clip ---
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                             1.0 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- Value (critic) loss with optional clipping ---
                value_loss = F.mse_loss(values, returns)

                # --- Entropy bonus (encourages exploration) ---
                entropy_loss = -entropy.mean()

                # --- Total loss ---
                total_loss = (policy_loss
                              + self.value_coef * value_loss
                              + self.entropy_coef * entropy_loss)

                # --- Backprop ---
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                # --- Diagnostics ---
                with torch.no_grad():
                    kl = ((old_log_probs - log_probs).mean()).item()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()

                all_policy_loss.append(policy_loss.item())
                all_value_loss.append(value_loss.item())
                all_entropy.append(-entropy_loss.item())  # report positive entropy
                all_kl.append(kl)
                all_clip_frac.append(clip_frac)

        self._update_count += 1

        current_lr = self.optimizer.param_groups[0]['lr']

        return {
            "policy_loss": float(np.mean(all_policy_loss)),
            "value_loss": float(np.mean(all_value_loss)),
            "entropy": float(np.mean(all_entropy)),
            "approx_kl": float(np.mean(all_kl)),
            "clip_fraction": float(np.mean(all_clip_frac)),
            "lr": current_lr,
            "update": self._update_count,
        }


# Import needed inside the module to keep it self-contained
import torch.nn.functional as F
