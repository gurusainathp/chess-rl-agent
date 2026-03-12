"""
src/opponents/opponent_pool.py
-------------------------------
Weighted sampler over a pool of chess opponents.

Why this matters
----------------
Pure self-play has a well-known failure mode: the model finds a local
equilibrium — a set of opening lines and responses — and stops exploring.
Both sides reinforce the same patterns, gradients collapse, and training
stalls.  The opponent pool breaks this by introducing diverse pressure:

  • RandomAgent        (10%) — forces the model to handle unfamiliar
                               positions it would never reach in self-play.
  • CheckpointAgents   (20%) — stable, non-updating targets from earlier
                               training.  The model must improve enough to
                               beat its past self.
  • Self-play          (60%) — keeps the primary signal intact.  The model
                               still learns primarily from itself.
  • Stockfish          (10%) — reserved for later; wired in as a no-op slot
                               today so the interface never changes.

Default weights
---------------
    OpponentPool.DEFAULT_WEIGHTS = {
        "self"       : 0.60,
        "checkpoint" : 0.20,
        "random"     : 0.10,
        "stockfish"  : 0.10,   # falls back to random if SF not available
    }

Checkpoint management
---------------------
Call pool.refresh_checkpoints() after each checkpoint save during training.
The pool scans the checkpoint directory and keeps the N most recent files
(default: pool_size=5) as CheckpointAgents.  If fewer than N checkpoints
exist the checkpoint weight is redistributed proportionally to self-play
so the ratios stay valid.

Usage
-----
    pool = OpponentPool(
        checkpoint_dir = "data/models",
        pool_size      = 5,
    )
    pool.refresh_checkpoints()

    opponent = pool.sample()     # returns None (self-play) or an agent
"""

from __future__ import annotations

import os
import random
import glob
from typing import Any

from src.opponents.checkpoint_agent import CheckpointAgent
from src.evaluation.random_agent    import RandomAgent


# ---------------------------------------------------------------------------
# Sentinel for "self-play" slot — no opponent object needed
# ---------------------------------------------------------------------------

_SELF_PLAY = None   # run_game treats opponent=None as pure self-play


class OpponentPool:
    """
    Weighted pool of opponents sampled each game.

    Parameters
    ----------
    checkpoint_dir : str
        Directory to scan for .pt checkpoint files.
    pool_size      : int
        Maximum number of CheckpointAgents to keep loaded (default 5).
        Older checkpoints beyond this window are discarded from the pool.
    weights        : dict[str, float] | None
        Override default sampling weights.  Keys must include
        'self', 'checkpoint', 'random', 'stockfish'.
        Values are normalised automatically so they need not sum to 1.
    random_seed    : int | None
        Seed for the pool's internal RNG (useful for reproducible tests).
    checkpoint_temperature : float
        Temperature used by all CheckpointAgents (default 0.5).
    device         : str
        Torch device for CheckpointAgents (default 'cpu').

    Methods
    -------
    refresh_checkpoints()
        Scan checkpoint_dir and update the loaded CheckpointAgent list.
        Call this after each checkpoint save during training.
    sample() -> agent | None
        Draw one opponent according to the pool weights.
        Returns None to signal self-play.
    summary() -> str
        Human-readable description of the current pool state.
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "self"       : 0.60,
        "checkpoint" : 0.20,
        "random"     : 0.10,
        "stockfish"  : 0.10,   # falls back to random until SF is wired in
    }

    def __init__(
        self,
        checkpoint_dir           : str,
        pool_size                : int   = 5,
        weights                  : dict  | None = None,
        random_seed              : int   | None = None,
        checkpoint_temperature   : float = 0.5,
        device                   : str   = "cpu",
    ):
        self.checkpoint_dir         = checkpoint_dir
        self.pool_size              = pool_size
        self.checkpoint_temperature = checkpoint_temperature
        self.device                 = device
        self._rng                   = random.Random(random_seed)

        # Merge caller weights with defaults
        raw = dict(self.DEFAULT_WEIGHTS)
        if weights:
            raw.update(weights)
        self._base_weights = raw

        # RandomAgent reused across all games (stateless)
        self._random_agent = RandomAgent()

        # CheckpointAgents — populated by refresh_checkpoints()
        self._checkpoint_agents: list[CheckpointAgent] = []

        # Perform an initial scan so the pool is ready immediately
        self.refresh_checkpoints()

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def refresh_checkpoints(self) -> None:
        """
        Scan checkpoint_dir for .pt files and (re)load the N most recent
        as CheckpointAgents.

        Safe to call at any point during training — existing agents whose
        files are still in the newest-N set are kept as-is (not reloaded).
        """
        if not os.path.isdir(self.checkpoint_dir):
            self._checkpoint_agents = []
            return

        # Find all .pt files in the directory, sorted newest-first by name
        # (epoch number is zero-padded in the filename so lexical sort works)
        pattern = os.path.join(self.checkpoint_dir, "policy_epoch_*.pt")
        paths   = sorted(glob.glob(pattern), reverse=True)   # newest first
        paths   = paths[:self.pool_size]                      # keep top N

        # Build a set of paths already loaded to avoid redundant disk reads
        loaded_paths = {a.path for a in self._checkpoint_agents}

        new_agents: list[CheckpointAgent] = []
        for path in paths:
            if path in loaded_paths:
                # Reuse the already-loaded agent
                existing = next(a for a in self._checkpoint_agents if a.path == path)
                new_agents.append(existing)
            else:
                try:
                    agent = CheckpointAgent(
                        path        = path,
                        temperature = self.checkpoint_temperature,
                        device      = self.device,
                    )
                    new_agents.append(agent)
                except Exception as exc:
                    # Never crash training due to a corrupt checkpoint
                    print(f"  [OpponentPool] Warning: could not load {path}: {exc}")

        self._checkpoint_agents = new_agents

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _effective_weights(self) -> dict[str, float]:
        """
        Compute the effective sampling weights given the current pool state.

        If no checkpoints are loaded, the checkpoint weight is folded into
        self-play so the distribution stays valid.
        """
        w = dict(self._base_weights)

        # Fold stockfish into random for now (not yet wired)
        w["random"] += w.pop("stockfish", 0.0)

        # If no checkpoints available, fold their weight into self-play
        if not self._checkpoint_agents:
            w["self"] += w.pop("checkpoint", 0.0)
        
        # Normalise
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}

    def sample(self) -> Any:
        """
        Sample one opponent according to the effective pool weights.

        Returns
        -------
        None                — signals pure self-play to run_game()
        RandomAgent         — uniform random opponent
        CheckpointAgent     — frozen snapshot of an earlier model
        """
        w = self._effective_weights()

        # Build a weighted list for random.choices
        slots:   list[Any]   = []
        weights: list[float] = []

        slots.append(_SELF_PLAY);          weights.append(w.get("self",       0.0))
        slots.append(self._random_agent);  weights.append(w.get("random",     0.0))

        if self._checkpoint_agents:
            # Pick one checkpoint agent uniformly at random for this slot
            cp_agent = self._rng.choice(self._checkpoint_agents)
            slots.append(cp_agent);        weights.append(w.get("checkpoint", 0.0))

        chosen = self._rng.choices(slots, weights=weights, k=1)[0]
        return chosen

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def n_checkpoints(self) -> int:
        """Number of checkpoint agents currently loaded."""
        return len(self._checkpoint_agents)

    def summary(self) -> str:
        w = self._effective_weights()
        cp_info = (
            f"{self.n_checkpoints} loaded "
            f"(epochs: {[a.epoch for a in self._checkpoint_agents]})"
            if self._checkpoint_agents else "none loaded yet"
        )
        return (
            f"OpponentPool | "
            f"self={w.get('self',0)*100:.0f}%  "
            f"checkpoint={w.get('checkpoint',0)*100:.0f}%  "
            f"random={w.get('random',0)*100:.0f}% | "
            f"Checkpoints: {cp_info}"
        )

    def __repr__(self) -> str:
        return f"OpponentPool(n_checkpoints={self.n_checkpoints})"