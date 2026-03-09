"""
train_policy.py
---------------
Training loop for the chess policy network.

Each epoch:
  1. Generate self-play games with the current model.
  2. Flatten all game samples into a training dataset.
  3. Compute cross-entropy loss over the chosen moves.
  4. Backpropagate and update model weights with Adam.
  5. Log metrics and optionally save a checkpoint.

Loss formulation
----------------
For each position in the dataset:

    logits = model(board_tensor, legal_moves)   # (1, num_legal_moves)
    loss   = CrossEntropyLoss(logits, target_move_index)

The target is simply the index of the move the model chose during
self-play.  This trains the network to assign higher probability to moves
that were selected in winning games (via the reward-weighted loss described
below) and lower probability to moves from losing games.

Reward-weighted loss
--------------------
Rather than treating all samples equally, each sample's loss contribution
is scaled by its reward:

    weighted_loss = loss * |reward|

This ensures:
  - Wins (+1)  : push strongly toward the chosen move
  - Losses (-1): push strongly AWAY from the chosen move (loss is negated)
  - Draws (0)  : sample contributes nothing to the gradient

Usage
-----
    python scripts/train.py

Or import directly:

    from src.training.train_policy import train, TrainingConfig

    config = TrainingConfig(n_epochs=10, games_per_epoch=20)
    model  = train(config)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.policy_network import PolicyNetwork
from src.training.self_play import run_games, records_to_dataset, GameSample


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """
    All training hyperparameters in one place.

    Attributes
    ----------
    n_epochs        : int   Number of training epochs.
    games_per_epoch : int   Self-play games generated per epoch.
    learning_rate   : float Adam learning rate.
    max_moves       : int   Per-game move cap passed to self-play.
    temp_high       : float Early-game sampling temperature.
    temp_low        : float Late-game sampling temperature.
    temp_threshold  : int   Move number at which temp drops to temp_low.
    checkpoint_dir  : str   Directory to save model checkpoints.
    checkpoint_every: int   Save a checkpoint every N epochs (0 = never).
    device          : str   Torch device ('cpu' or 'cuda').
    verbose         : bool  Print per-game self-play output.
    """
    n_epochs         : int   = 50
    games_per_epoch  : int   = 20
    learning_rate    : float = 1e-3
    max_moves        : int   = 200
    temp_high        : float = 1.0
    temp_low         : float = 0.1
    temp_threshold   : int   = 30
    checkpoint_dir   : str   = "data/models"
    checkpoint_every : int   = 10
    device           : str   = "cpu"
    verbose          : bool  = False


# ---------------------------------------------------------------------------
# Epoch metrics container
# ---------------------------------------------------------------------------

@dataclass
class EpochMetrics:
    """Metrics recorded at the end of each epoch."""
    epoch          : int
    loss           : float
    n_samples      : int
    n_games        : int
    wins           : int
    losses         : int
    draws          : int
    max_moves_hit  : int
    duration_sec   : float

    def summary(self) -> str:
        w = self.wins
        l = self.losses
        d = self.draws
        m = self.max_moves_hit
        g = self.n_games
        return (
            f"Epoch {self.epoch:>3} | "
            f"Loss: {self.loss:.4f} | "
            f"Samples: {self.n_samples:>5} | "
            f"W/L/D/Cap: {w}/{l}/{d}/{m} of {g} | "
            f"Time: {self.duration_sec:.1f}s"
        )


# ---------------------------------------------------------------------------
# Single training step over one batch of samples
# ---------------------------------------------------------------------------

def compute_loss(
    model   : PolicyNetwork,
    samples : list[GameSample],
    device  : str = "cpu",
) -> torch.Tensor:
    """
    Compute reward-weighted cross-entropy loss over a list of GameSamples.

    For each sample:
      - Forward pass produces logits over legal moves.
      - Cross-entropy loss is computed against the chosen move index.
      - Loss is scaled by |reward| and negated for losing moves.

    Samples with reward = 0 (draws) contribute nothing to the gradient.

    Parameters
    ----------
    model   : PolicyNetwork  (must be in train mode)
    samples : list[GameSample]
    device  : str

    Returns
    -------
    torch.Tensor  — scalar loss, ready for .backward()
    """
    # Zero-dim scalar so the final shape is () not (1,).
    # requires_grad=False here is intentional — grad_fn is attached the
    # moment we add a ce_loss (which does have a grad_fn) to it.
    weighted_loss = torch.zeros((), device=device)

    for sample in samples:
        # Skip samples with zero reward — they provide no learning signal
        if sample.reward == 0.0:
            continue

        # Prepare inputs
        board_t = sample.board_tensor.unsqueeze(0).to(device)   # (1, 13, 8, 8)
        target  = torch.tensor([sample.move_index], dtype=torch.long, device=device)

        # Forward pass → logits (1, num_legal_moves)
        logits = model(board_t, sample.legal_moves)

        # Per-sample cross-entropy loss
        ce_loss = F.cross_entropy(logits, target)

        # Scale by reward:
        #   Win  (+1): reinforce the chosen move   → add positive loss
        #   Loss (-1): suppress the chosen move    → negate loss (push away)
        weighted_loss = weighted_loss + ce_loss * sample.reward

    # Average over all non-zero-reward samples
    n_contributing = sum(1 for s in samples if s.reward != 0.0)
    if n_contributing > 0:
        weighted_loss = weighted_loss / n_contributing

    return weighted_loss


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model    : PolicyNetwork,
    epoch    : int,
    metrics  : EpochMetrics,
    save_dir : str,
) -> str:
    """
    Save model weights and epoch metadata to disk.

    Parameters
    ----------
    model    : PolicyNetwork
    epoch    : int
    metrics  : EpochMetrics
    save_dir : str  Directory to write into (created if absent).

    Returns
    -------
    str  Path to the saved checkpoint file.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"policy_epoch_{epoch:04d}.pt")

    torch.save(
        {
            "epoch"       : epoch,
            "model_state" : model.state_dict(),
            "metrics"     : {
                "loss"    : metrics.loss,
                "samples" : metrics.n_samples,
                "games"   : metrics.n_games,
            },
        },
        path,
    )
    return path


def load_checkpoint(path: str, model: PolicyNetwork) -> int:
    """
    Load model weights from a checkpoint file.

    Parameters
    ----------
    path  : str              Path to a .pt checkpoint file.
    model : PolicyNetwork    Model instance to load weights into.

    Returns
    -------
    int  The epoch number stored in the checkpoint.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint["epoch"]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    config : TrainingConfig | None = None,
    model  : PolicyNetwork  | None = None,
) -> tuple[PolicyNetwork, list[EpochMetrics]]:
    """
    Run the full training loop.

    Parameters
    ----------
    config : TrainingConfig
        Hyperparameters.  Defaults to TrainingConfig() if not provided.
    model : PolicyNetwork
        Model to train.  A fresh PolicyNetwork() is created if not provided.

    Returns
    -------
    model   : PolicyNetwork      — trained model (weights updated in place).
    history : list[EpochMetrics] — one entry per epoch.
    """
    if config is None:
        config = TrainingConfig()
    if model is None:
        model = PolicyNetwork()

    model.to(config.device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: list[EpochMetrics] = []

    print(f"\n{'='*60}")
    print(f"  Chess RL — Policy Network Training")
    print(f"{'='*60}")
    print(f"  Epochs          : {config.n_epochs}")
    print(f"  Games / epoch   : {config.games_per_epoch}")
    print(f"  Learning rate   : {config.learning_rate}")
    print(f"  Device          : {config.device}")
    print(f"{'='*60}\n")

    for epoch in range(1, config.n_epochs + 1):
        epoch_start = time.time()

        # ------------------------------------------------------------------
        # 1. Generate self-play games
        # ------------------------------------------------------------------
        model.eval()
        records = run_games(
            model,
            n_games        = config.games_per_epoch,
            max_moves      = config.max_moves,
            temp_high      = config.temp_high,
            temp_low       = config.temp_low,
            temp_threshold = config.temp_threshold,
            device         = config.device,
            verbose        = config.verbose,
        )

        # Tally game outcomes
        wins      = sum(1 for r in records if r.result == "white_wins")
        losses    = sum(1 for r in records if r.result == "black_wins")
        draws     = sum(1 for r in records if r.result == "draw")
        max_hit   = sum(1 for r in records if r.result == "max_moves_reached")

        # ------------------------------------------------------------------
        # 2. Build flat dataset
        # ------------------------------------------------------------------
        samples = records_to_dataset(records)

        if not samples:
            print(f"Epoch {epoch}: no samples generated — skipping.")
            continue

        # ------------------------------------------------------------------
        # 3. Compute loss and update weights
        # ------------------------------------------------------------------
        model.train()
        optimiser.zero_grad()

        loss = compute_loss(model, samples, device=config.device)

        # Guard: if every sample had reward=0 (all draws / capped games),
        # compute_loss returns a plain zero scalar — no grad_fn attached.
        # Calling .backward() on it raises RuntimeError, so skip the update.
        if loss.requires_grad:
            loss.backward()
            optimiser.step()

        # ------------------------------------------------------------------
        # 4. Record metrics
        # ------------------------------------------------------------------
        duration = time.time() - epoch_start
        metrics  = EpochMetrics(
            epoch         = epoch,
            loss          = loss.item(),
            n_samples     = len(samples),
            n_games       = len(records),
            wins          = wins,
            losses        = losses,
            draws         = draws,
            max_moves_hit = max_hit,
            duration_sec  = duration,
        )
        history.append(metrics)
        print(metrics.summary())

        # ------------------------------------------------------------------
        # 5. Save checkpoint
        # ------------------------------------------------------------------
        if config.checkpoint_every > 0 and epoch % config.checkpoint_every == 0:
            path = save_checkpoint(model, epoch, metrics, config.checkpoint_dir)
            print(f"  → Checkpoint saved: {path}")

    print(f"\nTraining complete. Total epochs: {len(history)}")
    return model, history