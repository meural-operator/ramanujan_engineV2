"""
Enhanced Actor-Critic Network for GCF Discovery.
Research-grade neural policy-value architecture with:
  - Residual connections + LayerNorm for better gradient flow
  - Orthogonal initialization for stable early training
  - Log-std clamping to prevent action distribution collapse/explosion
  - Separate value head with normalized output
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple


class ResidualBlock(nn.Module):
    """Single residual block with LayerNorm and GELU activation."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class ActorCriticGCFNetwork(nn.Module):
    """
    Neural Policy-Value network (AlphaZero / PPO style) for GCF convergence search.
    
    Observes the normalized GCF trajectory [prev_q_log, prev_p_log, q_log, p_log]
    and predicts:
      - Actor: continuous action distribution over polynomial coefficient sub-spaces
      - Critic: expected cumulative reward (digit-match value)
    
    Architecture:
      Input → Linear embedding → 2× Residual blocks (shared trunk)
                ├──▶ Actor head: mean + log_std (clamped to [-4, 2])
                └──▶ Critic head: scalar value
    """

    def __init__(self, state_dim: int = 4, hidden_dim: int = 256, action_dim: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Input embedding
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Shared residual trunk
        self.trunk = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )

        # Actor head: outputs mean and log_std for continuous Normal distribution
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        # Learnable log_std as a parameter (not output of network, avoids correlated noise)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)  # initial std ≈ 0.6

        # Critic head: scalar value estimate
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Orthogonal initialization for all linear layers
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
        # Actor mean should have smaller initial output to not immediately saturate
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        # Value head should also start small
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state: Tensor of shape (batch, state_dim) — log-normalized GCF trajectory
        
        Returns:
            action_mean:  (batch, action_dim) — distribution center
            action_std:   (batch, action_dim) — distribution spread (always > 0)
            value:        (batch, 1)          — critic value estimate
        """
        x = self.input_proj(state)
        x = self.trunk(x)

        # Actor
        action_mean = self.actor_mean(x)
        # Clamp log_std to [-4, 2] → std in [0.018, 7.4], prevents collapse/explosion
        log_std = self.actor_log_std.clamp(-4.0, 2.0).expand_as(action_mean)
        action_std = log_std.exp()

        # Critic
        value = self.critic(x)

        return action_mean, action_std, value

    def get_action_distribution(self, state: torch.Tensor) -> Normal:
        """Returns the action distribution for a given state (useful in PPO updates)."""
        mean, std, _ = self.forward(state)
        return Normal(mean, std)

    def evaluate_actions(self, states: torch.Tensor,
                          actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates log-probability of taken actions + entropy + value estimate.
        Used in the PPO surrogate loss computation.
        
        Returns:
            log_prob: (batch,)   - log π(a|s) under current policy
            entropy:  (batch,)   - distribution entropy for exploration bonus
            value:    (batch, 1) - critic estimate
        """
        mean, std, value = self.forward(states)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)  # sum over action dims
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value
