"""
random_agent.py
---------------
A random chess agent that selects uniformly from all legal moves.

This is the baseline opponent for the evaluation system.  A policy network
that cannot consistently beat a random agent has learned nothing meaningful,
so this provides the **minimum bar** the model must clear before moving to
stronger benchmarks (older checkpoints, Stockfish).

Usage
-----
    import chess
    from src.evaluation.random_agent import RandomAgent

    agent = RandomAgent(seed=42)   # seed for reproducibility in tests
    board = chess.Board()
    move  = agent.select_move(board)
    board.push(move)
"""

import random
import chess


class RandomAgent:
    """
    Chess agent that picks a move uniformly at random from all legal moves.

    Parameters
    ----------
    seed : int | None
        Optional random seed for reproducibility.  Pass None (default)
        for non-deterministic play during real evaluation runs.

    Methods
    -------
    select_move(board) → chess.Move
        Return a randomly chosen legal move from the current position.
    reset()
        Reset the internal RNG to its initial seed (useful between games
        when a fixed seed is set).
    """

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self._rng = random.Random(seed)

    def select_move(self, board: chess.Board) -> chess.Move:
        """
        Select a random legal move from the given board position.

        Parameters
        ----------
        board : chess.Board
            The current game state.  Must not be in a terminal position
            (i.e. board.is_game_over() must be False).

        Returns
        -------
        chess.Move

        Raises
        ------
        ValueError
            If there are no legal moves (game is already over).
        """
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError(
                "No legal moves available — board.is_game_over() should be "
                "checked before calling select_move()."
            )
        return self._rng.choice(legal)

    def reset(self) -> None:
        """Re-seed the internal RNG to its original seed value."""
        self._rng = random.Random(self.seed)

    def __repr__(self) -> str:
        return f"RandomAgent(seed={self.seed})"