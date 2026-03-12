"""
src/opponents/checkpoint_agent.py
----------------------------------
An agent that plays using a frozen snapshot of the policy network loaded
from a checkpoint file.

Used in the opponent pool to give the current model a non-trivial opponent
that represents an earlier version of itself.  Playing against past selves:

  • Breaks the symmetry collapse of pure self-play.
  • Provides a stable target — old checkpoints don't change as training
    progresses, unlike the live model.
  • Creates natural curriculum pressure: the current model must beat
    progressively older (weaker) versions of itself.

Interface
---------
    agent = CheckpointAgent("models/policy_epoch_0050.pt")
    move  = agent.select_move(board)   # same signature as RandomAgent

The agent is read-only — it never updates its weights during training.
"""

from __future__ import annotations

import os
import chess
import torch

from src.environment.board_encoder import encode_board
from src.models.policy_network import PolicyNetwork


class CheckpointAgent:
    """
    A frozen policy-network agent loaded from a checkpoint file.

    Parameters
    ----------
    path        : str    Path to a .pt checkpoint saved by train_policy.py.
    temperature : float  Sampling temperature (default 0.5 — slightly greedy
                         so the frozen opponent plays consistently, but not
                         fully deterministic).
    device      : str    Torch device (default 'cpu').

    Attributes
    ----------
    path        : str    Original checkpoint path.
    epoch       : int    Epoch number stored in the checkpoint.
    temperature : float

    Methods
    -------
    select_move(board) → chess.Move
    """

    def __init__(
        self,
        path        : str,
        temperature : float = 0.5,
        device      : str   = "cpu",
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        self.path        = path
        self.temperature = temperature
        self.device      = device

        # Load weights into a fresh PolicyNetwork and freeze it.
        self._model = PolicyNetwork()
        checkpoint  = torch.load(path, map_location="cpu")
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.to(device)
        self._model.eval()

        # Freeze all parameters — this agent must never be updated.
        for p in self._model.parameters():
            p.requires_grad_(False)

        self.epoch = checkpoint.get("epoch", -1)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def select_move(self, board: chess.Board) -> chess.Move:
        """
        Select a move using the frozen checkpoint model.

        Parameters
        ----------
        board : chess.Board  Current position (must not be game-over).

        Returns
        -------
        chess.Move
        """
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError(
                "No legal moves — check board.is_game_over() before calling "
                "select_move()."
            )

        state_t = torch.tensor(
            encode_board(board), dtype=torch.float32, device=self.device
        ).unsqueeze(0)   # (1, 13, 8, 8)

        with torch.no_grad():
            return self._model.select_move(state_t, legal, temperature=self.temperature)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CheckpointAgent("
            f"epoch={self.epoch}, "
            f"temperature={self.temperature}, "
            f"path='{os.path.basename(self.path)}')"
        )