"""
replay_buffer.py
----------------
Experience replay for the Chess RL policy network.

Why replay?
-----------
Without replay, each training epoch only sees the N games just played.
Early training is noisy — most games are capped draws that give no gradient
signal.  By storing a rolling window of positions and sampling uniformly
from it, we:

  • Re-use every position multiple times instead of once.
  • Stabilise gradient updates — each batch mixes old and new experience
    so a single bad epoch cannot overwrite all prior learning.
  • Increase effective batch diversity without playing more games.

Design
------
    buffer = ReplayBuffer(capacity=100_000)
    buffer.add(new_samples)        # add this epoch's samples
    batch  = buffer.sample(2_048)  # random draw for training
    print(buffer.summary())

The buffer is a plain deque(maxlen=capacity).  When full, the oldest
positions are automatically evicted as new ones arrive, keeping the
distribution roughly on-policy over recent checkpoints.

Capacity guidelines
-------------------
  60 games/epoch × ~80 half-moves ≈ 4 800 samples/epoch
  50 000 capacity ≈ last ~10 epochs   (fast turnover, more on-policy)
  100 000 capacity ≈ last ~20 epochs  ← default
  200 000 capacity ≈ last ~40 epochs  (more stable, more off-policy)
"""

from __future__ import annotations

import random
from collections import deque


class ReplayBuffer:
    """
    Fixed-capacity circular buffer of GameSample training tuples.

    Parameters
    ----------
    capacity : int
        Maximum positions to store (default 100 000).
        Oldest entries evicted automatically when full (FIFO).

    Attributes
    ----------
    capacity : int
    n_added  : int  Lifetime total of positions added (never resets).

    Methods
    -------
    add(samples)    Extend the buffer with a list of GameSamples.
    sample(n)       Return n random samples.
    clear()         Empty the buffer.
    is_ready        True when at least one sample exists.
    fill_ratio      Float in [0, 1] — how full the buffer is.
    summary()       One-line status string for logging.
    reward_stats()  Dict with reward distribution (useful for monitoring).
    """

    def __init__(self, capacity: int = 100_000) -> None:
        if capacity < 1:
            raise ValueError(f"ReplayBuffer capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)
        self.n_added: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add(self, samples: list) -> None:
        """
        Append a list of GameSamples to the buffer.

        When the buffer is full the oldest positions are silently dropped.

        Parameters
        ----------
        samples : list[GameSample]
        """
        self._buffer.extend(samples)
        self.n_added += len(samples)

    def sample(self, n: int) -> list:
        """
        Draw n samples uniformly at random.

        n <= len(buffer) : without replacement — no duplicates in the batch.
        n >  len(buffer) : with replacement    — so callers never get an
                           empty list even before the buffer is full.

        Parameters
        ----------
        n : int  Number of samples to draw.

        Returns
        -------
        list[GameSample]  (empty list if buffer is empty or n <= 0)
        """
        if n <= 0 or not self._buffer:
            return []
        buf = list(self._buffer)
        if n <= len(buf):
            return random.sample(buf, n)
        return random.choices(buf, k=n)   # with replacement when n > len

    def clear(self) -> None:
        """Remove all samples.  n_added is preserved for bookkeeping."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """True when at least one sample is stored."""
        return len(self._buffer) > 0

    @property
    def fill_ratio(self) -> float:
        """Fraction of capacity used, in [0.0, 1.0]."""
        return len(self._buffer) / self.capacity

    def reward_stats(self) -> dict:
        """
        Snapshot of the reward distribution currently in the buffer.

        Useful for diagnosing whether the buffer is dominated by draws
        (reward=0) early in training.

        Returns
        -------
        dict
            n_positive, n_negative, n_zero : int
            pct_nonzero                    : float  (percentage with |r| > 0)
        """
        n_pos = n_neg = n_zer = 0
        for s in self._buffer:
            r = s.reward
            if r > 0:
                n_pos += 1
            elif r < 0:
                n_neg += 1
            else:
                n_zer += 1
        total   = max(len(self._buffer), 1)
        pct_nz  = 100.0 * (n_pos + n_neg) / total
        return {
            "n_positive" : n_pos,
            "n_negative" : n_neg,
            "n_zero"     : n_zer,
            "pct_nonzero": round(pct_nz, 1),
        }

    def summary(self) -> str:
        """
        Human-readable one-liner for log output.

        Example
        -------
        ReplayBuffer   47,320 / 100,000  (47.3% full)  non-zero rewards: 18.4%
        """
        stats = self.reward_stats()
        return (
            f"ReplayBuffer  {len(self._buffer):>7,} / {self.capacity:,}"
            f"  ({self.fill_ratio * 100:.1f}% full)"
            f"  non-zero: {stats['pct_nonzero']:.1f}%"
        )