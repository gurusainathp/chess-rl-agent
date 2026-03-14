"""
stockfish_agent.py
------------------
A depth-limited Stockfish opponent for use in the training pool.

Design goals
------------
• Weak enough for an untrained model to occasionally beat:
  depth 4–6 plays roughly at club level, not engine strength.
• Same select_move(board) interface as RandomAgent and CheckpointAgent,
  so it drops straight into OpponentPool without any changes there.
• One engine process per agent instance — opened lazily on first move,
  kept alive between moves in the same game, closed explicitly via
  close() or via the context-manager protocol.

Depth vs time
-------------
We use Limit(depth=N) rather than Limit(time=T) because:
  • Deterministic — same depth on any machine, no wall-clock variance.
  • Predictable strength — depth 4 ≈ 1200 ELO, depth 6 ≈ 1800 ELO.
  • No accidental "full strength" on fast hardware with a generous time budget.

Recommended depths
------------------
  depth  4  — beginner/intermediate player, model can beat early on
  depth  5  — solid club player              ← default
  depth  6  — strong club player
  depth 10+ — much stronger, only use after 100+ epochs of training

Usage
-----
    from src.opponents.stockfish_agent import StockfishAgent

    agent = StockfishAgent(depth=5)
    move  = agent.select_move(board)
    agent.close()   # always close when done

    # Or as a context manager:
    with StockfishAgent(depth=5) as agent:
        move = agent.select_move(board)

Wiring into OpponentPool
------------------------
If your OpponentPool currently folds the "stockfish" slot into random,
replace that fallback with:

    from src.opponents.stockfish_agent import StockfishAgent, stockfish_available
    if stockfish_available():
        sf_agent = StockfishAgent(depth=5)
    else:
        sf_agent = None   # pool.sample() falls back to random
"""

from __future__ import annotations

import os
import shutil

import chess
import chess.engine


# ---------------------------------------------------------------------------
# Path resolution (reuse the same lookup as self_play.py)
# ---------------------------------------------------------------------------

def _find_stockfish() -> str | None:
    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    return shutil.which("stockfish")


def stockfish_available() -> bool:
    """Return True if a Stockfish binary can be found on this machine."""
    return _find_stockfish() is not None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class StockfishAgent:
    """
    Depth-limited Stockfish chess agent.

    Parameters
    ----------
    depth : int
        UCI search depth (default 5).  Lower = weaker and faster.
    sf_path : str | None
        Explicit path to the Stockfish binary.  Auto-detected if None.

    Raises
    ------
    RuntimeError
        If Stockfish is not found at instantiation time.

    Methods
    -------
    select_move(board) -> chess.Move
        Return the best move at the configured depth.
    close()
        Terminate the engine process.  Always call this (or use context
        manager) to avoid orphaned Stockfish processes.
    """

    def __init__(self, depth: int = 5, sf_path: str | None = None) -> None:
        self.depth = depth
        self._sf_path = sf_path or _find_stockfish()
        if self._sf_path is None:
            raise RuntimeError(
                "Stockfish binary not found.  Install it and either add it to "
                "PATH or set the STOCKFISH_PATH environment variable."
            )
        self._engine: chess.engine.SimpleEngine | None = None

    # ------------------------------------------------------------------
    # Lazy engine lifecycle
    # ------------------------------------------------------------------

    def _get_engine(self) -> chess.engine.SimpleEngine:
        """Open the engine process on first call; reuse thereafter."""
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self._sf_path)
        return self._engine

    def close(self) -> None:
        """Terminate the Stockfish process.  Safe to call multiple times."""
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "StockfishAgent":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def select_move(self, board: chess.Board) -> chess.Move:
        """
        Return the best move found at self.depth plies of search.

        Falls back to a random legal move if the engine call fails for
        any reason (corrupt position, process crash, etc.) so the game
        can always continue.

        Parameters
        ----------
        board : chess.Board

        Returns
        -------
        chess.Move
        """
        try:
            engine = self._get_engine()
            result = engine.play(board, chess.engine.Limit(depth=self.depth))
            if result.move is not None:
                return result.move
        except Exception:
            # Engine failure — fall back to random so the game can finish.
            self.close()   # reset engine state for next call

        import random
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("select_move called on a terminal position.")
        return random.choice(legal)

    def __repr__(self) -> str:
        return f"StockfishAgent(depth={self.depth})"