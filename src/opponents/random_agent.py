"""
src/opponents/random_agent.py
------------------------------
Re-exports RandomAgent from src.evaluation.random_agent so every opponent
lives under the same src/opponents/ namespace and shares one interface:

    agent.select_move(board: chess.Board) -> chess.Move

The underlying implementation is not duplicated — we import it directly.
This means fixes to the original propagate here automatically.

Usage
-----
    from src.opponents.random_agent import RandomAgent

    agent = RandomAgent()
    move  = agent.select_move(board)
"""

# Single source of truth lives in src/evaluation/random_agent.py.
# Import it here so callers can use either path interchangeably.
from src.evaluation.random_agent import RandomAgent  # noqa: F401

__all__ = ["RandomAgent"]