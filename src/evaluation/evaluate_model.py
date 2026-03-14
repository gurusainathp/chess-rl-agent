"""
evaluate_model.py
-----------------
Multi-opponent evaluation pipeline for the Chess RL policy network.

Evaluation tiers
----------------
  Tier 1 — vs RandomAgent      (default: 100 games)
    Baseline — a model that cannot beat random has learned nothing.

  Tier 2 — vs CheckpointAgent  (default: 50 games, needs a .pt path)
    Measures improvement relative to a previous version of itself.
    Positive winrate here = the current model is stronger than N epochs ago.

  Tier 3 — vs StockfishAgent   (default: 20 games, needs Stockfish binary)
    Absolute strength benchmark.  Use depth 4–6 so the model can
    occasionally compete; full-strength Stockfish will crush it early.

Output
------
  • Console summary for every tier that ran.
  • JSON appended/updated at eval_log_path (one entry per epoch).

JSON format (eval_results.json)
--------------------------------
  {
    "epoch_0010": {
      "epoch": 10,
      "timestamp": "2026-03-13 14:22:05",
      "vs_random":     { "n_games":100, "wins":72, "losses":18, ... },
      "vs_checkpoint": { ... },   <- omitted if tier not run
      "vs_stockfish":  { ... }    <- omitted if tier not run
    },
    "epoch_0020": { ... }
  }

Usage
-----
    from src.evaluation.evaluate_model import evaluate_full, EvaluationConfig

    config = EvaluationConfig(
        vs_random_games    = 100,
        vs_checkpoint_path = "models/policy_epoch_0010.pt",
        vs_stockfish_games = 20,
        stockfish_depth    = 5,
        eval_log_path      = "logs/eval_results.json",
    )
    result = evaluate_full(model, config, epoch=20)
    print(result.summary())

Backward compat
---------------
The original evaluate() signature is fully preserved for code in
train_policy.py that uses EvaluationConfig(n_games=...).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
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
    Full configuration for a multi-opponent evaluation run.

    Tier 1 (vs Random)
    ------------------
    vs_random_games : int   Games vs RandomAgent (0 = skip). Default 100.

    Tier 2 (vs Checkpoint)
    ----------------------
    vs_checkpoint_games    : int        Games vs a past checkpoint (0 = skip).
    vs_checkpoint_path     : str|None   Path to .pt file. If None, tier skipped.
    checkpoint_temperature : float      Checkpoint model temperature (default 0.1).

    Tier 3 (vs Stockfish)
    ---------------------
    vs_stockfish_games : int   Games vs depth-limited Stockfish (0 = skip).
    stockfish_depth    : int   UCI search depth (default 5 ~ 1600 ELO).

    Common
    ------
    max_moves       : int
    temperature     : float   Model sampling temperature (default 0.1, near-greedy).
    alternate_sides : bool    Model alternates White/Black (default True).
    eval_log_path   : str|None  Append JSON results here.
    verbose         : bool    Print per-game lines.
    device          : str

    Backward-compat
    ---------------
    n_games      : int        Alias for vs_random_games (legacy callers).
    random_seed  : int|None
    save_path    : str|None   Legacy alias for eval_log_path.
    """
    # Tier 1
    vs_random_games         : int        = 100

    # Tier 2
    vs_checkpoint_games     : int        = 0
    vs_checkpoint_path      : str | None = None
    checkpoint_temperature  : float      = 0.1

    # Tier 3
    vs_stockfish_games      : int        = 0
    stockfish_depth         : int        = 5

    # Common
    max_moves               : int        = 200
    temperature             : float      = 0.1
    alternate_sides         : bool       = True
    eval_log_path           : str | None = None
    verbose                 : bool       = False
    device                  : str        = "cpu"

    # Legacy fields
    n_games                 : int        = 0     # if >0 overrides vs_random_games
    random_seed             : int | None = None
    save_path               : str | None = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GameResult:
    """Single game outcome, from the model's perspective."""
    game_number  : int
    model_colour : str
    outcome      : Literal["win", "loss", "draw", "max_moves"]
    n_moves      : int


@dataclass
class TierResult:
    """
    Aggregate stats for one evaluation tier.
    wins + losses + draws + max_moves_games == n_games always.
    """
    opponent_name   : str
    n_games         : int
    wins            : int
    losses          : int
    draws           : int
    max_moves_games : int
    avg_game_length : float
    duration_sec    : float
    games           : list[GameResult] = field(default_factory=list)

    @property
    def winrate(self) -> float:
        return self.wins / self.n_games if self.n_games else 0.0

    @property
    def lossrate(self) -> float:
        return self.losses / self.n_games if self.n_games else 0.0

    @property
    def drawrate(self) -> float:
        return (self.draws + self.max_moves_games) / self.n_games if self.n_games else 0.0

    def summary(self) -> str:
        bar = "-" * 54
        w   = self.winrate * 100
        l   = self.lossrate * 100
        d   = self.drawrate * 100
        return (
            f"\n{bar}\n"
            f"  vs {self.opponent_name}  ({self.n_games} games, {self.duration_sec:.1f}s)\n"
            f"{bar}\n"
            f"  Wins   : {self.wins:>4}   ({w:.1f}%)\n"
            f"  Losses : {self.losses:>4}   ({l:.1f}%)\n"
            f"  Draws  : {self.draws + self.max_moves_games:>4}   ({d:.1f}%)"
            f"  [{self.draws} natural + {self.max_moves_games} capped]\n"
            f"  Avg length : {self.avg_game_length:.1f} half-moves\n"
            f"{bar}"
        )

    def to_dict(self) -> dict:
        return {
            "n_games"        : self.n_games,
            "wins"           : self.wins,
            "losses"         : self.losses,
            "draws"          : self.draws,
            "max_moves"      : self.max_moves_games,
            "winrate"        : round(self.winrate,  4),
            "lossrate"       : round(self.lossrate, 4),
            "drawrate"       : round(self.drawrate, 4),
            "avg_game_length": round(self.avg_game_length, 1),
        }


@dataclass
class EvaluationResult:
    """
    Combined result across all evaluation tiers for one epoch.

    Backward-compat properties surface the vs_random tier at the top level
    so old code that reads result.wins / result.winrate still works.
    """
    epoch          : int | None        = None
    timestamp      : str               = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    vs_random      : TierResult | None = None
    vs_checkpoint  : TierResult | None = None
    vs_stockfish   : TierResult | None = None

    # ── Backward-compat proxies to vs_random ──────────────────────────
    def _r(self) -> TierResult:
        if self.vs_random is None:
            raise AttributeError("vs_random tier was not run in this evaluation.")
        return self.vs_random

    @property
    def wins(self)              -> int:   return self._r().wins
    @property
    def losses(self)            -> int:   return self._r().losses
    @property
    def draws(self)             -> int:   return self._r().draws
    @property
    def max_moves_games(self)   -> int:   return self._r().max_moves_games
    @property
    def total_games(self)       -> int:   return self._r().n_games
    @property
    def winrate(self)           -> float: return self._r().winrate
    @property
    def lossrate(self)          -> float: return self._r().lossrate
    @property
    def drawrate(self)          -> float: return self._r().drawrate
    @property
    def average_game_length(self) -> float: return self._r().avg_game_length

    # ── Reporting ──────────────────────────────────────────────────────

    def summary(self) -> str:
        header = f"\n{'='*54}"
        if self.epoch is not None:
            header += f"\n  Evaluation — Epoch {self.epoch}  ({self.timestamp})"
        else:
            header += f"\n  Evaluation  ({self.timestamp})"
        header += f"\n{'='*54}"
        parts = [header]
        for tier in (self.vs_random, self.vs_checkpoint, self.vs_stockfish):
            if tier is not None:
                parts.append(tier.summary())
        parts.append("")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        d: dict = {"epoch": self.epoch, "timestamp": self.timestamp}
        if self.vs_random     is not None: d["vs_random"]     = self.vs_random.to_dict()
        if self.vs_checkpoint is not None: d["vs_checkpoint"] = self.vs_checkpoint.to_dict()
        if self.vs_stockfish  is not None: d["vs_stockfish"]  = self.vs_stockfish.to_dict()
        return d

    def save_json(self, path: str, epoch: int | None = None) -> None:
        """
        Append/update this result in a persistent JSON file.

        If the file already exists its history is preserved; this epoch's
        entry is added (or overwritten if the same epoch is re-evaluated).

        Parameters
        ----------
        path  : str          Destination file path.
        epoch : int | None   Overrides self.epoch for the JSON key.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Load existing history
        existing: dict = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}

        ep  = epoch if epoch is not None else self.epoch
        key = f"epoch_{ep:04d}" if ep is not None else self.timestamp.replace(" ", "_")
        existing[key] = self.to_dict()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _play_one_game(
    model          : PolicyNetwork,
    opponent,
    model_is_white : bool,
    max_moves      : int,
    temperature    : float,
    device         : str,
) -> tuple[str, int]:
    """Play one game, return (outcome_from_model_pov, n_half_moves)."""
    board   = chess.Board()
    n_moves = 0

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
            ).unsqueeze(0)
            with torch.no_grad():
                move = model.select_move(state_t, legal, temperature=temperature)
        else:
            try:
                move = opponent.select_move(board)
            except Exception:
                import random as _r
                move = _r.choice(legal)

        board.push(move)
        n_moves += 1

    if not board.is_game_over():
        return "max_moves", n_moves

    res = board.result()
    if res == "1/2-1/2":
        return "draw", n_moves
    model_won = (res == "1-0" and model_is_white) or (res == "0-1" and not model_is_white)
    return ("win" if model_won else "loss"), n_moves


def _run_tier(
    model         : PolicyNetwork,
    opponent,
    opponent_name : str,
    n_games       : int,
    config        : EvaluationConfig,
) -> TierResult:
    """Run N games vs opponent and return a TierResult."""
    wins = losses = draws = max_moves_games = 0
    total_moves = 0
    game_results: list[GameResult] = []
    t0 = time.time()

    for i in range(n_games):
        model_is_white = (i % 2 == 0) if config.alternate_sides else True
        colour_str     = "white" if model_is_white else "black"

        outcome, n_moves = _play_one_game(
            model, opponent, model_is_white,
            config.max_moves, config.temperature, config.device,
        )

        if   outcome == "win":       wins            += 1
        elif outcome == "loss":      losses          += 1
        elif outcome == "draw":      draws           += 1
        else:                        max_moves_games += 1

        total_moves += n_moves
        game_results.append(GameResult(
            game_number=i + 1, model_colour=colour_str,
            outcome=outcome, n_moves=n_moves,
        ))

        if config.verbose:
            print(
                f"  [{opponent_name}] {i+1:>3}/{n_games} | "
                f"Model:{colour_str:<5} | {outcome:<9} | {n_moves} moves"
            )

    return TierResult(
        opponent_name   = opponent_name,
        n_games         = n_games,
        wins            = wins,
        losses          = losses,
        draws           = draws,
        max_moves_games = max_moves_games,
        avg_game_length = total_moves / n_games if n_games else 0.0,
        duration_sec    = time.time() - t0,
        games           = game_results,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_full(
    model  : PolicyNetwork,
    config : EvaluationConfig | None = None,
    epoch  : int | None = None,
) -> EvaluationResult:
    """
    Run all configured evaluation tiers and return combined results.

    Prints a full summary to console and optionally saves JSON.

    Parameters
    ----------
    model  : PolicyNetwork
    config : EvaluationConfig
    epoch  : int | None   Used as the JSON key (e.g. 10 → "epoch_0010").

    Returns
    -------
    EvaluationResult
    """
    if config is None:
        config = EvaluationConfig()

    model.eval()
    result = EvaluationResult(epoch=epoch)

    # Resolve legacy n_games field
    n_random = config.n_games if config.n_games > 0 else config.vs_random_games

    # ── Tier 1: vs RandomAgent ────────────────────────────────────────
    if n_random > 0:
        random_agent = RandomAgent(seed=config.random_seed)
        result.vs_random = _run_tier(
            model, random_agent, "RandomAgent", n_random, config
        )

    # ── Tier 2: vs Checkpoint ─────────────────────────────────────────
    if config.vs_checkpoint_games > 0:
        path = config.vs_checkpoint_path
        if path and os.path.exists(path):
            try:
                from src.training.train_policy import load_checkpoint
                ckpt_model = PolicyNetwork().to(config.device)
                load_checkpoint(path, ckpt_model)
                ckpt_model.eval()

                _temp = config.checkpoint_temperature
                _dev  = config.device

                class _CkptAgent:
                    def select_move(self, board):
                        legal = list(board.legal_moves)
                        st = torch.tensor(
                            encode_board(board), dtype=torch.float32, device=_dev
                        ).unsqueeze(0)
                        with torch.no_grad():
                            return ckpt_model.select_move(st, legal, temperature=_temp)

                ckpt_name = os.path.basename(path)
                result.vs_checkpoint = _run_tier(
                    model, _CkptAgent(),
                    f"Checkpoint({ckpt_name})",
                    config.vs_checkpoint_games, config,
                )
            except Exception as e:
                print(f"  [eval] Checkpoint tier skipped — {e}")
        else:
            if path:
                print(f"  [eval] Checkpoint not found: {path} — skipping tier 2")
            else:
                print("  [eval] vs_checkpoint_games > 0 but no vs_checkpoint_path — skipping tier 2")

    # ── Tier 3: vs Stockfish ──────────────────────────────────────────
    if config.vs_stockfish_games > 0:
        try:
            from src.opponents.stockfish_agent import StockfishAgent, stockfish_available
            if stockfish_available():
                with StockfishAgent(depth=config.stockfish_depth) as sf_agent:
                    result.vs_stockfish = _run_tier(
                        model, sf_agent,
                        f"Stockfish(depth={config.stockfish_depth})",
                        config.vs_stockfish_games, config,
                    )
            else:
                print("  [eval] Stockfish binary not found — skipping tier 3")
        except Exception as e:
            print(f"  [eval] Stockfish tier failed — {e}")

    # ── Console output ────────────────────────────────────────────────
    print(result.summary())

    # ── JSON logging ──────────────────────────────────────────────────
    log_path = config.eval_log_path or config.save_path
    if log_path:
        result.save_json(log_path, epoch=epoch)
        print(f"  [eval] Appended results → {log_path}")

    return result


def evaluate(
    model  : PolicyNetwork,
    config : EvaluationConfig | None = None,
) -> EvaluationResult:
    """
    Backward-compatible wrapper around evaluate_full().

    Called by train_policy.train() — runs whatever tiers are configured,
    defaulting to vs-random only if no checkpoint/stockfish options are set.
    """
    return evaluate_full(model, config, epoch=None)