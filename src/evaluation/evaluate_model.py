"""
evaluate_model.py
-----------------
Evaluation system for the chess policy network.

Runs the model against the RandomAgent over N games and reports results
from the **model's perspective** — wins, losses, and draws are all
attributed to the model regardless of which colour it played.

To remove colour bias, by default the model alternates sides:
  - Even-numbered games: model plays White
  - Odd-numbered  games: model plays Black

This ensures the winrate is not inflated by always having the first-move
advantage, and makes the evaluation metric more honest.

Result dataclass
----------------
EvaluationResult is returned from evaluate() and contains:
  - wins, losses, draws, total_games
  - winrate, drawrate, lossrate  (0.0–1.0)
  - average_game_length          (half-moves)
  - per_game breakdown           (list of GameResult)

It is also printed to console and optionally saved to JSON.

Usage
-----
    from src.models.policy_network import PolicyNetwork
    from src.evaluation.evaluate_model import evaluate, EvaluationConfig

    model  = PolicyNetwork()
    config = EvaluationConfig(n_games=100)
    result = evaluate(model, config)

    print(result.summary())
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Literal

import chess
import torch

from src.environment.board_encoder import encode_board
from src.evaluation.random_agent import RandomAgent
from src.models.policy_network import PolicyNetwork


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvaluationConfig:
    """
    Configuration for a model evaluation run.

    Attributes
    ----------
    n_games      : int   Total games to play (default: 100).
    max_moves    : int   Move cap per game — capped games count as draws
                         (default: 200).
    temperature  : float Sampling temperature for the model (default: 0.1).
                         Low temperature → near-greedy → best evaluation play.
    random_seed  : int | None  Seed for the random agent (default: None).
    alternate_sides : bool
                  If True (default), model alternates White/Black each game.
                  If False, model always plays White.
    save_path    : str | None  If set, save JSON results to this path.
    verbose      : bool  Print a one-line summary per game (default: False).
    device       : str   Torch device (default: 'cpu').
    """
    n_games         : int        = 100
    max_moves       : int        = 200
    temperature     : float      = 0.1
    random_seed     : int | None = None
    alternate_sides : bool       = True
    save_path       : str | None = None
    verbose         : bool       = False
    device          : str        = "cpu"


# ---------------------------------------------------------------------------
# Per-game result
# ---------------------------------------------------------------------------

@dataclass
class GameResult:
    """
    Result of a single evaluation game, always from the model's perspective.

    Attributes
    ----------
    game_number  : int
    model_colour : str   'white' or 'black'
    outcome      : str   'win', 'loss', 'draw', or 'max_moves'
    n_moves      : int   Number of half-moves played
    """
    game_number  : int
    model_colour : str
    outcome      : Literal["win", "loss", "draw", "max_moves"]
    n_moves      : int


# ---------------------------------------------------------------------------
# Aggregate result
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """
    Aggregate statistics from a full evaluation run.

    All rates are in [0.0, 1.0].  wins + losses + draws + max_moves_games
    always equals total_games.
    """
    total_games      : int
    wins             : int
    losses           : int
    draws            : int
    max_moves_games  : int
    average_game_length : float
    games            : list[GameResult] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived rates (computed properties, not stored fields)
    # ------------------------------------------------------------------

    @property
    def winrate(self) -> float:
        return self.wins / self.total_games if self.total_games > 0 else 0.0

    @property
    def lossrate(self) -> float:
        return self.losses / self.total_games if self.total_games > 0 else 0.0

    @property
    def drawrate(self) -> float:
        d = self.draws + self.max_moves_games
        return d / self.total_games if self.total_games > 0 else 0.0

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted multi-line summary string."""
        bar = "=" * 50
        return (
            f"\n{bar}\n"
            f"  Evaluation vs Random Agent ({self.total_games} games)\n"
            f"{bar}\n"
            f"  Wins   : {self.wins:>4}   ({self.winrate * 100:.1f}%)\n"
            f"  Losses : {self.losses:>4}   ({self.lossrate * 100:.1f}%)\n"
            f"  Draws  : {self.draws:>4}   ({self.drawrate * 100:.1f}%)\n"
            f"  Capped : {self.max_moves_games:>4}\n"
            f"\n"
            f"  Winrate          : {self.winrate * 100:.1f}%\n"
            f"  Avg game length  : {self.average_game_length:.1f} moves\n"
            f"{bar}\n"
        )

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for JSON saving)."""
        d = asdict(self)
        d["winrate"]   = self.winrate
        d["lossrate"]  = self.lossrate
        d["drawrate"]  = self.drawrate
        return d

    def save_json(self, path: str) -> None:
        """
        Save the evaluation result to a JSON file.

        Parameters
        ----------
        path : str  File path to write.  Parent directories are created
                    automatically.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------

def _play_one_game(
    model        : PolicyNetwork,
    random_agent : RandomAgent,
    model_is_white: bool,
    max_moves    : int,
    temperature  : float,
    device       : str,
) -> tuple[str, int]:
    """
    Play one game between the model and the random agent.

    Returns
    -------
    outcome : str   'win', 'loss', 'draw', or 'max_moves'
                    Always from the model's perspective.
    n_moves : int   Number of half-moves played.
    """
    board    = chess.Board()
    n_moves  = 0

    for _ in range(max_moves):
        if board.is_game_over():
            break

        legal = list(board.legal_moves)
        if not legal:
            break

        model_to_move = (board.turn == chess.WHITE) == model_is_white

        if model_to_move:
            state_t = torch.tensor(
                encode_board(board), dtype=torch.float32, device=device
            ).unsqueeze(0)                         # (1, 13, 8, 8)
            with torch.no_grad():
                move = model.select_move(state_t, legal, temperature=temperature)
        else:
            move = random_agent.select_move(board)

        board.push(move)
        n_moves += 1

    # ------------------------------------------------------------------
    # Determine outcome from the model's perspective
    # ------------------------------------------------------------------
    if not board.is_game_over():
        return "max_moves", n_moves

    result = board.result()   # "1-0", "0-1", or "1/2-1/2"

    if result == "1/2-1/2":
        return "draw", n_moves

    model_won = (result == "1-0" and model_is_white) or \
                (result == "0-1" and not model_is_white)

    return ("win" if model_won else "loss"), n_moves


# ---------------------------------------------------------------------------
# Public API: evaluate()
# ---------------------------------------------------------------------------

def evaluate(
    model  : PolicyNetwork,
    config : EvaluationConfig | None = None,
) -> EvaluationResult:
    """
    Evaluate the model against the RandomAgent over N games.

    Results are always reported from the **model's perspective**.
    By default the model alternates White/Black to avoid colour bias.

    Parameters
    ----------
    model  : PolicyNetwork
    config : EvaluationConfig   Defaults to EvaluationConfig() if None.

    Returns
    -------
    EvaluationResult
        Full statistics dataclass.  Also printed to console.
    """
    if config is None:
        config = EvaluationConfig()

    model.eval()
    random_agent = RandomAgent(seed=config.random_seed)

    wins = losses = draws = max_moves_games = 0
    total_moves = 0
    game_results: list[GameResult] = []

    start = time.time()

    for i in range(config.n_games):
        # Alternate sides to remove first-move bias
        model_is_white = (i % 2 == 0) if config.alternate_sides else True
        colour_str     = "white" if model_is_white else "black"

        outcome, n_moves = _play_one_game(
            model=model,
            random_agent=random_agent,
            model_is_white=model_is_white,
            max_moves=config.max_moves,
            temperature=config.temperature,
            device=config.device,
        )

        # Tally
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        elif outcome == "draw":
            draws += 1
        else:
            max_moves_games += 1

        total_moves += n_moves

        game_results.append(GameResult(
            game_number=i + 1,
            model_colour=colour_str,
            outcome=outcome,
            n_moves=n_moves,
        ))

        if config.verbose:
            print(
                f"  Game {i + 1:>4}/{config.n_games} | "
                f"Model: {colour_str:<5} | "
                f"Outcome: {outcome:<9} | "
                f"Moves: {n_moves}"
            )

    elapsed = time.time() - start
    avg_len = total_moves / config.n_games if config.n_games > 0 else 0.0

    result = EvaluationResult(
        total_games=config.n_games,
        wins=wins,
        losses=losses,
        draws=draws,
        max_moves_games=max_moves_games,
        average_game_length=avg_len,
        games=game_results,
    )

    # Always print to console
    print(result.summary())
    print(f"  Evaluation completed in {elapsed:.1f}s")

    # Optionally save to JSON
    if config.save_path:
        result.save_json(config.save_path)
        print(f"  Results saved to: {config.save_path}")

    return result