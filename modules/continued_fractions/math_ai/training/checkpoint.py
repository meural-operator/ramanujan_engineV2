"""
Checkpoint Manager for the GCF Actor-Critic network.
Handles saving / restoring training state, best-model tracking, and model export.
"""
import os
import json
import torch
from typing import Optional


class CheckpointManager:
    """
    Manages saving and loading of model + optimizer state for the PPO / MCTS trainer.
    
    Saves a checkpoint whenever the observed mean reward exceeds the historical best.
    Also writes a human-readable JSON metadata file alongside each checkpoint.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_mean_reward = -float('inf')
        self._meta_path = os.path.join(checkpoint_dir, "training_state.json")
        self._load_meta()

    def _load_meta(self):
        """Restore best_mean_reward from a previous training session if available."""
        if os.path.exists(self._meta_path):
            try:
                with open(self._meta_path, 'r') as f:
                    meta = json.load(f)
                    self.best_mean_reward = float(meta.get('best_mean_reward', -float('inf')))
                    print(f"[Checkpoint] Resuming — historical best reward: {self.best_mean_reward:.4f}")
            except Exception:
                pass

    def _save_meta(self, episode: int, mean_reward: float, extra: dict = None):
        meta = {
            "episode": episode,
            "best_mean_reward": self.best_mean_reward,
            "last_mean_reward": mean_reward,
        }
        if extra:
            meta.update(extra)
        with open(self._meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             episode: int, mean_reward: float, filename: str = "em_mcts.pt",
             extra_meta: dict = None) -> bool:
        """
        Saves checkpoint if mean_reward > best_mean_reward.
        
        Returns:
            True if a new best checkpoint was saved, False otherwise.
        """
        if mean_reward <= self.best_mean_reward:
            return False

        self.best_mean_reward = mean_reward
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'episode': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_mean_reward': self.best_mean_reward,
        }, path)
        self._save_meta(episode, mean_reward, extra_meta)
        print(f"[Checkpoint] ✅ NEW BEST saved @ ep {episode:,} | reward={mean_reward:.4f} → {path}")
        return True

    def save_always(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    episode: int, mean_reward: float, filename: str = "em_mcts_latest.pt"):
        """Unconditionally saves (used for periodic heartbeat checkpoints)."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'episode': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_mean_reward': self.best_mean_reward,
        }, path)

    @staticmethod
    def load(path: str, model: torch.nn.Module,
             optimizer: Optional[torch.optim.Optimizer] = None,
             device: torch.device = None) -> dict:
        """
        Loads checkpoint from disk.
        
        Args:
            path: Path to the .pt checkpoint file.
            model: Network instance to restore weights into.
            optimizer: If provided, also restores optimizer state.
            device: Target device (defaults to CPU if checkpoint was on GPU and CUDA is unavailable).
        
        Returns:
            The full checkpoint dict (contains 'episode', 'best_mean_reward', etc.).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        map_location = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(path, map_location=map_location, weights_only=False)

        model.load_state_dict(ckpt['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        print(f"[Checkpoint] Loaded from {path} (episode {ckpt.get('episode', '?')}, "
              f"best reward {ckpt.get('best_mean_reward', '?'):.4f})")
        return ckpt

    def latest_path(self, filename: str = "em_mcts.pt") -> str:
        """Returns absolute path to the best checkpoint (may not exist yet)."""
        return os.path.join(self.checkpoint_dir, filename)

    def exists(self, filename: str = "em_mcts.pt") -> bool:
        return os.path.exists(os.path.join(self.checkpoint_dir, filename))
