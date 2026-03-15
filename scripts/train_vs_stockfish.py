"""
scripts/train_vs_stockfish.py
------------------------------
Stockfish curriculum trainer for the Chess RL policy network.

The model trains exclusively against Stockfish, starting at depth 1
and automatically promoting to deeper levels as it proves mastery.

Curriculum progression
----------------------
  Depth 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8  (configurable max)

Promotion condition
-------------------
The model is promoted to the next depth when its rolling average win rate
over the last ``--promotion-window`` evaluations reaches ``--promotion-threshold``
(default 60%).

Example: with window=3 and threshold=0.60, the model must score ≥ 60%
across the last 3 evaluation runs before advancing.

On promotion
-------------
  • A promotion checkpoint is saved.
  • The replay buffer is kept (carry-over experience helps at the next depth).
  • Training continues immediately at the new depth with a fresh StockfishAgent.
  • A clear banner is printed in the log.

On reaching max depth
---------------------
Training continues at max depth until ``--epochs`` is exhausted.

Core components reused from the main training pipeline
-------------------------------------------------------
  compute_loss()   — reward-weighted REINFORCE with baseline normalisation
  ReplayBuffer     — experience replay
  setup_logger()   — dual console + file logging
  save_checkpoint() / load_checkpoint()
  run_game()       — single game with opening randomisation
  records_to_dataset()
  EpochMetrics     — metrics container

Reward schedule (same as main trainer)
---------------------------------------
  Win  = +2.0   model beat Stockfish
  Draw =  0.0   no signal
  Loss = -2.0   Stockfish beat model

Usage
-----
    # All defaults — starts at depth 1, promotes up to depth 8
    python scripts/train_vs_stockfish.py

    # Longer run with custom thresholds
    python scripts/train_vs_stockfish.py \\
        --epochs 500                   \\
        --games-per-epoch 30           \\
        --max-depth 5                  \\
        --promotion-threshold 0.60     \\
        --promotion-window 3           \\
        --eval-games 50                \\
        --eval-every 5                 \\
        --checkpoint-dir models/sf_curriculum \\
        --log-dir logs

    # Resume from a checkpoint at a specific depth
    python scripts/train_vs_stockfish.py \\
        --resume models/sf_curriculum/policy_epoch_0050.pt \\
        --start-depth 3

    # Disable auto-promotion (train at a fixed depth)
    python scripts/train_vs_stockfish.py --start-depth 3 --max-depth 3
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from src.environment.board_encoder import encode_board
from src.models.policy_network import PolicyNetwork
from src.opponents.stockfish_agent import StockfishAgent, stockfish_available
from src.training.replay_buffer import ReplayBuffer
from src.training.self_play import run_game, records_to_dataset, GameRecord
from src.training.train_policy import (
    compute_loss,
    setup_logger,
    save_checkpoint,
    load_checkpoint,
    EpochMetrics,
)

import chess


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SFCurriculumConfig:
    """
    All hyperparameters for the Stockfish curriculum trainer.

    Curriculum
    ----------
    start_depth         : int    Stockfish depth to begin at (default 1).
    max_depth           : int    Never promote beyond this depth (default 8).
    promotion_threshold : float  Win rate needed to promote (default 0.60).
    promotion_window    : int    Number of consecutive evals used for rolling
                                 average (default 3).  Model must sustain the
                                 threshold across this many evals.

    Training
    ---------
    n_epochs            : int
    games_per_epoch     : int
    learning_rate       : float
    entropy_coeff       : float
    max_moves           : int
    temp_high           : float
    temp_low            : float
    temp_threshold      : int
    opening_moves       : int
    opening_prob        : float

    Replay buffer
    -------------
    use_replay_buffer   : bool
    replay_capacity     : int
    replay_batch_size   : int

    Evaluation
    ----------
    eval_every          : int    Evaluate every N epochs (also checks promotion).
    eval_games          : int    Games per evaluation run.

    Checkpointing / logging
    -----------------------
    checkpoint_dir      : str
    checkpoint_every    : int
    log_dir             : str
    device              : str
    verbose             : bool
    pgn_dir             : str | None
    pgn_every           : int
    """
    # Curriculum
    start_depth          : int   = 1
    max_depth            : int   = 5
    promotion_threshold  : float = 0.60
    promotion_window     : int   = 3

    # Training
    n_epochs             : int   = 300
    games_per_epoch      : int   = 20
    learning_rate        : float = 1e-3
    entropy_coeff        : float = 0.02
    max_moves            : int   = 60
    temp_high            : float = 1.2
    temp_low             : float = 0.1
    temp_threshold       : int   = 30
    opening_moves        : int   = 4
    opening_prob         : float = 1.0

    # Replay buffer
    use_replay_buffer    : bool  = True
    replay_capacity      : int   = 200_000
    replay_batch_size    : int   = 2_048

    # Evaluation
    eval_every           : int   = 5
    eval_games           : int   = 50

    # Checkpointing / logging
    checkpoint_dir       : str   = "models/sf_curriculum"
    checkpoint_every     : int   = 10
    log_dir              : str   = "logs"
    device               : str   = "cpu"
    verbose              : bool  = False
    pgn_dir              : str | None = None
    pgn_every            : int   = 10


# ---------------------------------------------------------------------------
# Promotion tracker
# ---------------------------------------------------------------------------

class PromotionTracker:
    """
    Tracks win rates over a rolling window of evaluations and signals
    when the promotion threshold has been sustained.

    Parameters
    ----------
    threshold : float   Win rate (0–1) needed to promote.
    window    : int     Number of consecutive eval results to average over.
    """

    def __init__(self, threshold: float, window: int) -> None:
        self.threshold = threshold
        self.window    = window
        self._history: deque[float] = deque(maxlen=window)

    def record(self, win_rate: float) -> None:
        """Add the win rate from the latest evaluation."""
        self._history.append(win_rate)

    @property
    def rolling_win_rate(self) -> float:
        """Mean win rate over the current window (0.0 if no data yet)."""
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    @property
    def ready_to_promote(self) -> bool:
        """
        True when the window is full AND the rolling average meets threshold.
        Requires all ``window`` slots to be filled — prevents early promotion
        after just one lucky evaluation.
        """
        return (
            len(self._history) == self.window
            and self.rolling_win_rate >= self.threshold
        )

    def reset(self) -> None:
        """Clear history after a promotion."""
        self._history.clear()

    def summary(self) -> str:
        filled = len(self._history)
        rates  = [f"{r*100:.1f}%" for r in self._history]
        avg    = f"{self.rolling_win_rate*100:.1f}%"
        return (
            f"PromotionTracker  window={filled}/{self.window}  "
            f"rates=[{', '.join(rates)}]  avg={avg}  "
            f"threshold={self.threshold*100:.0f}%  "
            f"ready={'YES ✓' if self.ready_to_promote else 'no'}"
        )


# ---------------------------------------------------------------------------
# Evaluation vs Stockfish at a specific depth
# ---------------------------------------------------------------------------

def evaluate_vs_stockfish(
    model      : PolicyNetwork,
    depth      : int,
    n_games    : int,
    max_moves  : int,
    temperature: float,
    device     : str,
    log,
) -> float:
    """
    Play n_games against Stockfish at the given depth.
    Model alternates colours.  Returns win rate (wins / n_games).

    Draws and losses both count as non-wins for promotion purposes —
    the model must actually beat Stockfish, not just draw.

    Returns
    -------
    float  Win rate in [0, 1].
    """
    log.info(f"  [Eval] vs Stockfish depth={depth}  ({n_games} games) ...")

    wins = losses = draws = caps = 0

    with StockfishAgent(depth=depth) as sf:
        for i in range(n_games):
            model_is_white = (i % 2 == 0)
            model_color    = chess.WHITE if model_is_white else chess.BLACK

            record = run_game(
                model,
                opponent       = sf,
                model_color    = model_color,
                max_moves      = max_moves,
                temp_high      = temperature,
                temp_low       = temperature,
                temp_threshold = 0,          # fixed temperature for eval
                device         = device,
                use_stockfish  = False,      # don't call SF twice for cap eval
                opening_moves  = 0,          # no randomisation in eval
                opening_prob   = 0.0,
            )

            # Outcome from model's perspective
            if record.result == "white_wins":
                if model_is_white: wins += 1
                else:              losses += 1
            elif record.result == "black_wins":
                if not model_is_white: wins += 1
                else:                  losses += 1
            elif record.result in ("max_moves_reached", "max_moves_draw"):
                caps += 1
            else:
                draws += 1

    win_rate = wins / n_games if n_games else 0.0

    col_w = 7
    sep   = "  " + "-" * 54
    log.info(sep)
    log.info(
        f"  {'Opponent':<20} {'Games':>{col_w}} {'Win':>{col_w}} "
        f"{'Loss':>{col_w}} {'Draw':>{col_w}} {'Cap':>{col_w}}"
    )
    log.info(sep)
    log.info(
        f"  {f'Stockfish(d={depth})':<20} "
        f"{n_games:>{col_w}} {wins:>{col_w}} {losses:>{col_w}} "
        f"{draws:>{col_w}} {caps:>{col_w}}"
    )
    log.info(sep)
    log.info(f"  Win rate: {win_rate*100:.1f}%  ({wins}/{n_games} wins)")
    log.info(sep)

    return win_rate


# ---------------------------------------------------------------------------
# Per-epoch game runner (model vs Stockfish only, with opening randomisation)
# ---------------------------------------------------------------------------

def run_sf_games(
    model         : PolicyNetwork,
    sf_agent      : StockfishAgent,
    n_games       : int,
    config        : SFCurriculumConfig,
) -> list[GameRecord]:
    """
    Run n_games where the model always faces Stockfish.
    Model alternates White/Black each game.
    Returns a list of GameRecords.
    Always prints a per-epoch summary.
    """
    records: list[GameRecord] = []
    wins = losses = draws = cap_win = cap_loss = cap_draw = 0

    for i in range(n_games):
        model_color = chess.WHITE if i % 2 == 0 else chess.BLACK

        record = run_game(
            model,
            opponent       = sf_agent,
            model_color    = model_color,
            max_moves      = config.max_moves,
            temp_high      = config.temp_high,
            temp_low       = config.temp_low,
            temp_threshold = config.temp_threshold,
            device         = config.device,
            use_stockfish  = True,
            opening_moves  = config.opening_moves,
            opening_prob   = config.opening_prob,
        )
        record.opponent_type = f"StockfishAgent(d={sf_agent.depth})"
        records.append(record)

        model_is_white = (model_color == chess.WHITE)

        if record.result == "white_wins":
            if model_is_white: wins += 1
            else:              losses += 1
        elif record.result == "black_wins":
            if not model_is_white: wins += 1
            else:                  losses += 1
        elif record.result == "max_moves_reached":
            cp = record.stockfish_cp
            from src.training.self_play import STOCKFISH_WIN_THRESHOLD_CP
            if cp is not None and cp >= STOCKFISH_WIN_THRESHOLD_CP:
                if model_is_white: cap_win += 1
                else:              cap_loss += 1
            elif cp is not None and cp <= -STOCKFISH_WIN_THRESHOLD_CP:
                if not model_is_white: cap_win += 1
                else:                  cap_loss += 1
            else:
                cap_draw += 1
        else:
            draws += 1

        if config.verbose:
            color_str = "W" if model_color == chess.WHITE else "B"
            sf_str = (f"  sf={record.stockfish_cp:+.0f}cp"
                      if record.stockfish_cp is not None else "")
            print(
                f"  Game {i+1:>3}/{n_games} | "
                f"Model:{color_str} | "
                f"Moves:{record.n_moves:>3} | "
                f"Result:{record.result}{sf_str}"
            )

    # Per-epoch summary
    col_w = 7
    sep   = "  " + "-" * 68
    opp   = f"Stockfish(d={sf_agent.depth})"
    print(f"  Games this epoch: {n_games}")
    print(sep)
    print(
        f"  {'Opponent':<22} {'Games':>{col_w}} {'Win':>{col_w}} "
        f"{'Loss':>{col_w}} {'Draw':>{col_w}} "
        f"{'Cap-W':>{col_w}} {'Cap-L':>{col_w}} {'Cap-D':>{col_w}}"
    )
    print(sep)
    print(
        f"  {opp:<22} "
        f"{n_games:>{col_w}} {wins:>{col_w}} {losses:>{col_w}} "
        f"{draws:>{col_w}} {cap_win:>{col_w}} {cap_loss:>{col_w}} {cap_draw:>{col_w}}"
    )
    print(sep)

    return records


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_vs_stockfish(
    config      : SFCurriculumConfig | None = None,
    model       : PolicyNetwork | None = None,
    start_epoch : int = 0,
    start_depth : int | None = None,
) -> tuple[PolicyNetwork, list[EpochMetrics]]:
    """
    Run the Stockfish curriculum training loop.

    Parameters
    ----------
    config      : SFCurriculumConfig
    model       : PolicyNetwork  (fresh one created if None)
    start_epoch : int            Epoch to resume from.
    start_depth : int | None     Override config.start_depth (for CLI resume).

    Returns
    -------
    model   : PolicyNetwork
    history : list[EpochMetrics]
    """
    if config is None:
        config = SFCurriculumConfig()
    if model is None:
        model = PolicyNetwork()

    model.to(config.device)

    # ── Logger ────────────────────────────────────────────────────────────
    log = setup_logger(config.log_dir)

    # ── Stockfish availability check ──────────────────────────────────────
    if not stockfish_available():
        log.error(
            "Stockfish binary not found.\n"
            "Install Stockfish and either add it to PATH or set "
            "the STOCKFISH_PATH environment variable."
        )
        sys.exit(1)

    current_depth = start_depth if start_depth is not None else config.start_depth
    optimiser     = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history       : list[EpochMetrics] = []
    train_start   = time.time()
    total_epochs  = start_epoch + config.n_epochs
    resume_str    = f" (resuming from epoch {start_epoch})" if start_epoch > 0 else ""

    # ── Banner ────────────────────────────────────────────────────────────
    W = 64
    log.info("=" * W)
    log.info(f"  Chess RL — Stockfish Curriculum Trainer{resume_str}")
    log.info("=" * W)

    log.info("  GENERAL")
    log.info(f"    device              : {config.device}")
    log.info(f"    epochs to run       : {config.n_epochs}  (up to epoch {total_epochs})")
    log.info(f"    games / epoch       : {config.games_per_epoch}")
    log.info(f"    log dir             : {config.log_dir}")
    log.info(f"    checkpoint dir      : {config.checkpoint_dir}")
    log.info(f"    checkpoint every    : {config.checkpoint_every} epochs")

    log.info("  CURRICULUM")
    log.info(f"    start depth         : {current_depth}")
    log.info(f"    max depth           : {config.max_depth}")
    log.info(f"    promotion threshold : {config.promotion_threshold*100:.0f}% win rate")
    log.info(f"    promotion window    : {config.promotion_window} consecutive evals")
    log.info(f"    eval every          : {config.eval_every} epochs")
    log.info(f"    eval games          : {config.eval_games} games per evaluation")

    log.info("  TRAINING")
    log.info(f"    learning rate       : {config.learning_rate}")
    log.info(f"    entropy coeff       : {config.entropy_coeff}")
    log.info( "    reward schedule     : Win=+2.0  Draw= 0.0  Loss=-2.0  (baseline-normalised)")

    log.info("  TEMPERATURE  (Phase 1)")
    log.info(f"    temp high           : {config.temp_high}  (moves 0–{config.temp_threshold-1})")
    log.info(f"    temp low            : {config.temp_low}  (moves {config.temp_threshold}+)")
    log.info(f"    max moves / game    : {config.max_moves}")

    log.info("  OPENING RANDOMISATION")
    log.info(f"    opening moves       : {config.opening_moves}  (0 = disabled)")
    log.info(f"    opening prob        : {config.opening_prob*100:.0f}% of games")

    log.info("  REPLAY BUFFER")
    log.info(f"    enabled             : {config.use_replay_buffer}")
    if config.use_replay_buffer:
        log.info(f"    capacity            : {config.replay_capacity:,} positions")
        log.info(f"    batch size          : {config.replay_batch_size:,} positions / step")

    log.info("=" * W)
    log.info("")

    # ── Replay buffer ─────────────────────────────────────────────────────
    replay: ReplayBuffer | None = None
    if config.use_replay_buffer:
        replay = ReplayBuffer(capacity=config.replay_capacity)
        log.info(f"  {replay.summary()}\n")

    # ── Promotion tracker ─────────────────────────────────────────────────
    tracker = PromotionTracker(
        threshold = config.promotion_threshold,
        window    = config.promotion_window,
    )

    # ── Stockfish agent at current depth ──────────────────────────────────
    sf_agent = StockfishAgent(depth=current_depth)
    log.info(f"  Starting at Stockfish depth {current_depth}\n")

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, total_epochs + 1):
        epoch_start = time.time()

        log.info(f"\n{'─'*64}")
        log.info(f"  Epoch {epoch}/{total_epochs}  |  Stockfish depth={current_depth}")
        log.info(f"{'─'*64}")

        # 1. Generate games ------------------------------------------------
        model.eval()
        records = run_sf_games(
            model     = model,
            sf_agent  = sf_agent,
            n_games   = config.games_per_epoch,
            config    = config,
        )

        # 2. Replay buffer + dataset --------------------------------------
        new_samples = records_to_dataset(records)
        if not new_samples:
            log.info("  No samples generated — skipping epoch.")
            continue

        if replay is not None:
            replay.add(new_samples)
            samples = replay.sample(config.replay_batch_size)
            log.info(f"  {replay.summary()}")
        else:
            samples = new_samples

        # 3. Compute loss and update --------------------------------------
        model.train()
        optimiser.zero_grad()
        loss = compute_loss(
            model, samples,
            device        = config.device,
            entropy_coeff = config.entropy_coeff,
        )

        # Health metrics
        n_nonzero = sum(1 for s in samples if s.reward != 0.0)
        pct_nz    = 100.0 * n_nonzero / len(samples) if samples else 0.0

        mean_ent = 0.0
        if samples:
            model.eval()
            with torch.no_grad():
                probe    = random.sample(samples, min(200, len(samples)))
                ent_vals = []
                for s in probe:
                    bt = s.board_tensor.unsqueeze(0).to(config.device)
                    lg = model(bt, s.legal_moves)
                    lp = F.log_softmax(lg, dim=-1)
                    ent_vals.append(-(lp.exp() * lp).sum().item())
            mean_ent = sum(ent_vals) / len(ent_vals)
            model.train()

        if loss.requires_grad:
            loss.backward()
            optimiser.step()
            log.info(f"  Training loss     : {loss.item():.6f}")
        else:
            log.info("  Training loss     : 0.000000  (no update — all rewards zero)")

        # 4. Record metrics -----------------------------------------------
        wins    = sum(1 for r in records if r.result == "white_wins")
        losses  = sum(1 for r in records if r.result == "black_wins")
        draws   = sum(1 for r in records if r.result == "draw")
        cap_dec = sum(1 for r in records if r.result == "max_moves_reached")
        cap_drw = sum(1 for r in records if r.result == "max_moves_draw")
        sf_used = sum(1 for r in records if r.stockfish_cp is not None)
        avg_len = sum(r.n_moves for r in records) / len(records) if records else 0.0

        log.info(f"  Games this epoch  : {len(records)}")
        log.info(f"  New samples       : {len(new_samples)}")
        log.info(f"  Training batch    : {len(samples)}"
                 + (" (from replay)" if replay is not None else ""))
        log.info(f"  Stockfish evals   : {sf_used}")
        log.info(f"  Avg game length   : {avg_len:.1f} half-moves")
        log.info(f"  Non-zero rewards  : {n_nonzero}/{len(samples)}  ({pct_nz:.1f}%)"
                 + ("  ✓" if 40 <= pct_nz <= 70 else "  ⚠ target 40–70%"))
        log.info(f"  Mean entropy      : {mean_ent:.4f}"
                 + ("  ✓" if mean_ent >= 0.5 else "  ⚠ target ≥ 0.5 (low = collapsing)"))

        duration = time.time() - epoch_start
        metrics  = EpochMetrics(
            epoch         = epoch,
            loss          = loss.item(),
            n_samples     = len(samples),
            n_games       = len(records),
            wins          = wins,
            losses        = losses,
            draws         = draws,
            max_moves_hit = cap_dec + cap_drw,
            duration_sec  = duration,
        )
        history.append(metrics)
        log.info(metrics.summary())

        # 5. Checkpoint ---------------------------------------------------
        if config.checkpoint_every > 0 and epoch % config.checkpoint_every == 0:
            path = save_checkpoint(model, epoch, metrics, config.checkpoint_dir)
            log.info(f"  → Checkpoint saved: {path}")

        # 6. PGN saving ---------------------------------------------------
        if (config.pgn_dir and config.pgn_every > 0
                and epoch % config.pgn_every == 0):
            try:
                from src.training.pgn_writer import records_to_pgn, save_pgn
                pgn_text = records_to_pgn(records, epoch=epoch)
                pgn_path = save_pgn(pgn_text, epoch=epoch, pgn_dir=config.pgn_dir)
                log.info(f"  → PGN saved: {pgn_path}")
            except Exception as e:
                log.info(f"  → PGN save failed: {e}")

        # 7. Evaluation + promotion check ---------------------------------
        run_eval = (
            config.eval_every > 0 and epoch % config.eval_every == 0
        ) or (epoch == total_epochs)

        if run_eval:
            model.eval()
            win_rate = evaluate_vs_stockfish(
                model       = model,
                depth       = current_depth,
                n_games     = config.eval_games,
                max_moves   = config.max_moves,
                temperature = config.temp_low,
                device      = config.device,
                log         = log,
            )
            model.train()

            tracker.record(win_rate)
            log.info(f"  {tracker.summary()}")

            # ── Promotion check ──────────────────────────────────────────
            if tracker.ready_to_promote and current_depth < config.max_depth:
                new_depth = current_depth + 1

                log.info("")
                log.info("  " + "★" * 60)
                log.info(f"  ★  PROMOTION: depth {current_depth} → {new_depth}")
                log.info(f"  ★  Rolling win rate {tracker.rolling_win_rate*100:.1f}% ≥ "
                         f"{config.promotion_threshold*100:.0f}% over "
                         f"{config.promotion_window} evals")
                log.info("  " + "★" * 60)
                log.info("")

                # Save a promotion checkpoint
                promo_path = os.path.join(
                    config.checkpoint_dir,
                    f"promotion_depth{current_depth}_to_{new_depth}_epoch{epoch:04d}.pt"
                )
                save_checkpoint(model, epoch, metrics, config.checkpoint_dir)
                log.info(f"  → Promotion checkpoint saved: {promo_path}")

                # Swap Stockfish agent to new depth
                sf_agent.close()
                current_depth = new_depth
                sf_agent      = StockfishAgent(depth=current_depth)

                # Reset tracker (fresh window at new depth)
                tracker.reset()
                log.info(f"  Continuing at depth {current_depth} — promotion window reset\n")

            elif current_depth == config.max_depth:
                log.info(
                    f"  At max depth ({config.max_depth}).  "
                    f"Continuing to train.  Rolling win rate: "
                    f"{tracker.rolling_win_rate*100:.1f}%"
                )

        log.info(f"  Epoch time        : {duration:.1f}s")

    # ── Final summary ─────────────────────────────────────────────────────
    sf_agent.close()
    total_time = time.time() - train_start

    log.info(f"\n{'='*64}")
    log.info(f"  Curriculum training complete")
    log.info(f"  Total time        : {total_time:.1f}s  ({total_time/60:.1f} min)")
    log.info(f"  Epochs trained    : {len(history)}")
    log.info(f"  Final depth       : {current_depth}")
    log.info(f"  Final win rate    : {tracker.rolling_win_rate*100:.1f}%  (rolling)")
    log.info(f"{'='*64}\n")

    return model, history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the Chess RL policy network via Stockfish curriculum.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Curriculum ─────────────────────────────────────────────────────────
    g = p.add_argument_group("Curriculum")
    g.add_argument("--start-depth",           type=int,   default=1,
                   help="Stockfish depth to start training at.")
    g.add_argument("--max-depth",             type=int,   default=5,
                   help="Maximum Stockfish depth to promote to.")
    g.add_argument("--promotion-threshold",   type=float, default=0.60,
                   help="Win rate needed to advance to next depth (0–1).")
    g.add_argument("--promotion-window",      type=int,   default=3,
                   help="Number of consecutive evaluations to average for promotion.")

    # ── Training duration ──────────────────────────────────────────────────
    g = p.add_argument_group("Training duration")
    g.add_argument("--epochs",                type=int,   default=300,
                   help="Total training epochs.")
    g.add_argument("--games-per-epoch",       type=int,   default=20,
                   help="Games vs Stockfish per epoch.")

    # ── Optimiser ──────────────────────────────────────────────────────────
    g = p.add_argument_group("Optimiser")
    g.add_argument("--lr",                    type=float, default=1e-3,
                   help="Adam learning rate.")
    g.add_argument("--entropy-coeff",         type=float, default=0.02,
                   help="Entropy regularisation coefficient.")

    # ── Self-play / temperature ─────────────────────────────────────────────
    g = p.add_argument_group("Temperature")
    g.add_argument("--max-moves",             type=int,   default=60,
                   help="Move cap per game.")
    g.add_argument("--temp-high",             type=float, default=1.2,
                   help="Sampling temperature for early moves.")
    g.add_argument("--temp-low",              type=float, default=0.1,
                   help="Sampling temperature for late moves.")
    g.add_argument("--temp-threshold",        type=int,   default=30,
                   help="Half-move at which temperature drops.")

    # ── Opening randomisation ───────────────────────────────────────────────
    g = p.add_argument_group("Opening randomisation")
    g.add_argument("--opening-moves",         type=int,   default=4,
                   help="Random half-moves at game start (0 = disabled).")
    g.add_argument("--opening-prob",          type=float, default=1.0,
                   help="Fraction of games using opening randomisation.")

    # ── Replay buffer ───────────────────────────────────────────────────────
    g = p.add_argument_group("Replay buffer")
    g.add_argument("--no-replay",             action="store_true",
                   help="Disable replay buffer.")
    g.add_argument("--replay-capacity",       type=int,   default=200_000,
                   help="Max positions in replay buffer.")
    g.add_argument("--replay-batch-size",     type=int,   default=2_048,
                   help="Positions sampled per training step.")

    # ── Evaluation ──────────────────────────────────────────────────────────
    g = p.add_argument_group("Evaluation")
    g.add_argument("--eval-every",            type=int,   default=5,
                   help="Evaluate every N epochs (also checks promotion).")
    g.add_argument("--eval-games",            type=int,   default=50,
                   help="Games per evaluation run.")

    # ── Checkpointing ───────────────────────────────────────────────────────
    g = p.add_argument_group("Checkpointing")
    g.add_argument("--checkpoint-dir",        type=str,   default="models/sf_curriculum",
                   help="Directory to save checkpoints.")
    g.add_argument("--checkpoint-every",      type=int,   default=10,
                   help="Save a checkpoint every N epochs (0 = never).")
    g.add_argument("--resume",                type=str,   default=None,
                   help="Path to a .pt checkpoint file to resume from.")

    # ── PGN ─────────────────────────────────────────────────────────────────
    g = p.add_argument_group("PGN")
    g.add_argument("--pgn-dir",               type=str,   default=None,
                   help="Save game PGNs to this directory.")
    g.add_argument("--pgn-every",             type=int,   default=10,
                   help="Save PGN every N epochs.")

    # ── Hardware / misc ──────────────────────────────────────────────────────
    g = p.add_argument_group("Hardware / misc")
    g.add_argument("--device",                type=str,   default="cpu",
                   help="Torch device: 'cpu' or 'cuda'.")
    g.add_argument("--log-dir",               type=str,   default="logs",
                   help="Directory for training log files.")
    g.add_argument("--verbose",               action="store_true",
                   help="Print per-game output during self-play.")

    return p


def main() -> None:
    args = build_parser().parse_args()

    config = SFCurriculumConfig(
        # Curriculum
        start_depth          = args.start_depth,
        max_depth            = args.max_depth,
        promotion_threshold  = args.promotion_threshold,
        promotion_window     = args.promotion_window,
        # Training
        n_epochs             = args.epochs,
        games_per_epoch      = args.games_per_epoch,
        learning_rate        = args.lr,
        entropy_coeff        = args.entropy_coeff,
        max_moves            = args.max_moves,
        temp_high            = args.temp_high,
        temp_low             = args.temp_low,
        temp_threshold       = args.temp_threshold,
        opening_moves        = args.opening_moves,
        opening_prob         = args.opening_prob,
        # Replay buffer
        use_replay_buffer    = not args.no_replay,
        replay_capacity      = args.replay_capacity,
        replay_batch_size    = args.replay_batch_size,
        # Evaluation
        eval_every           = args.eval_every,
        eval_games           = args.eval_games,
        # Checkpointing / logging
        checkpoint_dir       = args.checkpoint_dir,
        checkpoint_every     = args.checkpoint_every,
        log_dir              = args.log_dir,
        device               = args.device,
        verbose              = args.verbose,
        pgn_dir              = args.pgn_dir,
        pgn_every            = args.pgn_every,
    )

    model       = PolicyNetwork()
    start_epoch = 0

    if args.resume:
        if not os.path.exists(args.resume):
            print(f"ERROR: checkpoint not found: {args.resume}")
            sys.exit(1)
        start_epoch = load_checkpoint(args.resume, model)
        print(f"Resumed from: {args.resume}  (epoch {start_epoch})")

    train_vs_stockfish(
        config      = config,
        model       = model,
        start_epoch = start_epoch,
        start_depth = args.start_depth,
    )


if __name__ == "__main__":
    main()