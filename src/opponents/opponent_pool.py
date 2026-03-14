"""
src/opponents/opponent_pool.py
-------------------------------
Weighted sampler over a pool of chess opponents.

Opponent slots
--------------
  self        (60%) — pure self-play; no opponent object
  checkpoint  (20%) — frozen past-self snapshots; builds a curriculum
  random      (10%) — RandomAgent; exposes the model to chaotic positions
  stockfish   (10%) — depth-limited StockfishAgent; punishes tactical errors
                      Falls back to RandomAgent if Stockfish is unavailable.

The pool is constructed once at training start and updated every time a
checkpoint is saved via refresh_checkpoints().

Stockfish fallback
------------------
If the Stockfish binary is not found the 10% stockfish weight is
transparently redistributed to the random slot.  A one-time warning is
printed so the user knows.  No crash, no config change needed.

Usage
-----
    pool = OpponentPool(checkpoint_dir="models", pool_size=5)
    pool.refresh_checkpoints()
    opponent = pool.sample()   # None = self-play, else an agent object
"""

from __future__ import annotations

import glob
import os
import random
from typing import Any

from src.opponents.checkpoint_agent import CheckpointAgent
from src.evaluation.random_agent    import RandomAgent


_SELF_PLAY = None   # run_game treats opponent=None as pure self-play


class OpponentPool:
    """
    Weighted pool of opponents sampled once per game.

    Parameters
    ----------
    checkpoint_dir         : str    Directory scanned for .pt checkpoint files.
    pool_size              : int    Max CheckpointAgents kept loaded (default 5).
    weights                : dict   Sampling weights for each slot.
                                    Keys: "self", "checkpoint", "random", "stockfish".
                                    Normalised automatically.
    random_seed            : int|None
    checkpoint_temperature : float  Temperature for CheckpointAgent inference (default 0.5).
    stockfish_depth        : int    UCI depth for StockfishAgent (default 5).
    device                 : str    Torch device for checkpoint agents.
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "self"       : 0.40,   # max 40% self-play
        "stockfish"  : 0.30,   # depth-8 Stockfish (falls back to random)
        "checkpoint" : 0.20,   # past checkpoints (falls back to stockfish)
        "random"     : 0.10,   # random agent
    }

    # Hard cap: self-play never exceeds 40% regardless of fallbacks
    SELF_PLAY_MAX: float = 0.40

    def __init__(
        self,
        checkpoint_dir           : str,
        pool_size                : int   = 5,
        weights                  : dict  | None = None,
        random_seed              : int   | None = None,
        checkpoint_temperature   : float = 0.5,
        stockfish_depth          : int   = 8,
        device                   : str   = "cpu",
    ):
        self.checkpoint_dir         = checkpoint_dir
        self.pool_size              = pool_size
        self.checkpoint_temperature = checkpoint_temperature
        self.stockfish_depth        = stockfish_depth
        self.device                 = device
        self._rng                   = random.Random(random_seed)

        raw = dict(self.DEFAULT_WEIGHTS)
        if weights:
            raw.update(weights)
        self._base_weights = raw

        # ── Agents ──────────────────────────────────────────────────────
        self._random_agent  = RandomAgent()
        self._stockfish_agent = None    # set in _init_stockfish()
        self._sf_available  = False
        self._sf_reason     = ""        # human-readable reason if unavailable
        self._init_stockfish()

        self._checkpoint_agents: list[CheckpointAgent] = []
        self.refresh_checkpoints()

    # ------------------------------------------------------------------
    # Stockfish initialisation
    # ------------------------------------------------------------------

    def _init_stockfish(self) -> None:
        """
        Try to load StockfishAgent once at construction time.

        Sets self._stockfish_agent and self._sf_available.
        On failure, stockfish weight folds into random — printed once.
        """
        try:
            from src.opponents.stockfish_agent import StockfishAgent, stockfish_available
            if stockfish_available():
                self._stockfish_agent = StockfishAgent(depth=self.stockfish_depth)
                self._sf_available    = True
                print(f"  [OpponentPool] Stockfish OK  "
                      f"(depth={self.stockfish_depth})  "
                      f"→ {self._base_weights.get('stockfish', 0.10)*100:.0f}% pool slot active")
            else:
                self._sf_available = False
                self._sf_reason    = "binary not found (set STOCKFISH_PATH)"
                print(f"  [OpponentPool] Stockfish unavailable — {self._sf_reason}\n"
                      f"                 stockfish slot ({self._base_weights.get('stockfish',0.10)*100:.0f}%)"
                      f" folded into random agent")
        except ImportError as e:
            self._sf_available = False
            self._sf_reason    = f"import error: {e}"
            print(f"  [OpponentPool] Stockfish unavailable — {self._sf_reason}\n"
                  f"                 stockfish slot folded into random agent")

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def refresh_checkpoints(self) -> None:
        """
        Scan checkpoint_dir and (re)load the N most recent as CheckpointAgents.
        Safe to call after every checkpoint save during training.
        """
        if not os.path.isdir(self.checkpoint_dir):
            self._checkpoint_agents = []
            return

        pattern = os.path.join(self.checkpoint_dir, "policy_epoch_*.pt")
        paths   = sorted(glob.glob(pattern), reverse=True)[:self.pool_size]

        loaded_paths = {a.path for a in self._checkpoint_agents}
        new_agents: list[CheckpointAgent] = []

        for path in paths:
            if path in loaded_paths:
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
                    print(f"  [OpponentPool] Warning: could not load {path}: {exc}")

        self._checkpoint_agents = new_agents

    # ------------------------------------------------------------------
    # Effective weights
    # ------------------------------------------------------------------

    def _effective_weights(self) -> dict[str, float]:
        """
        Compute live sampling weights given current pool state.

        Fallback rules (applied in order):
          1. If no checkpoints loaded → fold checkpoint weight into stockfish.
          2. If Stockfish unavailable → fold stockfish weight into random.
          3. Cap self-play at SELF_PLAY_MAX (40%) — excess goes to stockfish,
             or random if stockfish is also unavailable.
          4. Normalise so weights sum to 1.0.
        """
        w = dict(self._base_weights)

        # Rule 1: No checkpoints → fold into stockfish (not self-play)
        if not self._checkpoint_agents:
            w["stockfish"] = w.get("stockfish", 0.0) + w.pop("checkpoint", 0.0)

        # Rule 2: Stockfish unavailable → fold into random
        if not self._sf_available:
            w["random"] = w.get("random", 0.0) + w.pop("stockfish", 0.0)

        # Rule 3: Hard cap self-play at SELF_PLAY_MAX
        overflow = w.get("self", 0.0) - self.SELF_PLAY_MAX
        if overflow > 0:
            w["self"] = self.SELF_PLAY_MAX
            # Give overflow to stockfish if available, else random
            if self._sf_available:
                w["stockfish"] = w.get("stockfish", 0.0) + overflow
            else:
                w["random"] = w.get("random", 0.0) + overflow

        # Rule 4: Normalise
        total = sum(w.values()) or 1.0
        return {k: v / total for k, v in w.items()}

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self) -> Any:
        """
        Sample one opponent according to effective weights.

        Returns
        -------
        None            — self-play (run_game handles this)
        RandomAgent     — uniform random opponent
        CheckpointAgent — frozen past-self snapshot
        StockfishAgent  — depth-limited engine (if available)
        """
        w = self._effective_weights()

        slots:   list[Any]   = []
        weights: list[float] = []

        # Self-play slot
        slots.append(_SELF_PLAY)
        weights.append(w.get("self", 0.0))

        # Random slot (includes folded stockfish weight when SF unavailable)
        slots.append(self._random_agent)
        weights.append(w.get("random", 0.0))

        # Checkpoint slot
        if self._checkpoint_agents:
            cp_agent = self._rng.choice(self._checkpoint_agents)
            slots.append(cp_agent)
            weights.append(w.get("checkpoint", 0.0))

        # Stockfish slot (only added when actually available)
        if self._sf_available and self._stockfish_agent is not None:
            slots.append(self._stockfish_agent)
            weights.append(w.get("stockfish", 0.0))

        return self._rng.choices(slots, weights=weights, k=1)[0]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def n_checkpoints(self) -> int:
        return len(self._checkpoint_agents)

    def summary(self) -> str:
        w = self._effective_weights()

        sf_str = (
            f"depth={self.stockfish_depth}"
            if self._sf_available
            else f"UNAVAILABLE ({self._sf_reason}) → random"
        )
        cp_str = (
            f"{self.n_checkpoints} loaded "
            f"(epochs: {[a.epoch for a in self._checkpoint_agents]})"
            if self._checkpoint_agents
            else "none yet — weight folded into self-play"
        )

        return (
            f"OpponentPool\n"
            f"    self        {w.get('self',       0)*100:>5.1f}%\n"
            f"    checkpoint  {w.get('checkpoint', 0)*100:>5.1f}%  {cp_str}\n"
            f"    random      {w.get('random',     0)*100:>5.1f}%\n"
            f"    stockfish   {w.get('stockfish',  0)*100:>5.1f}%  {sf_str}"
        )

    def __repr__(self) -> str:
        return (f"OpponentPool("
                f"n_checkpoints={self.n_checkpoints}, "
                f"sf={'on' if self._sf_available else 'off'})")