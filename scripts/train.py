"""
scripts/train.py
----------------
Main training entry point for the Chess RL Agent.

Orchestrates the full training pipeline:
  1. Generate self-play games with the current policy
  2. Flatten game records into a training dataset
  3. Compute reward-weighted cross-entropy loss and update weights
  4. Save a checkpoint (configurable interval)
  5. Evaluate the model against the random agent (configurable interval)
  6. Print a per-epoch summary to the console

Usage
-----
    # Run with all defaults (50 epochs, 20 games/epoch)
    python scripts/train.py

    # Common overrides
    python scripts/train.py --epochs 100 --games-per-epoch 30 --lr 5e-4

    # Resume from a checkpoint
    python scripts/train.py --resume models/policy_epoch_0010.pt

    # Evaluate every 5 epochs, save checkpoints every 10
    python scripts/train.py --eval-every 5 --checkpoint-every 10

    # Full example (recommended overnight run)
    python scripts/train.py \\
        --epochs 150           \\
        --games-per-epoch 60   \\
        --lr 1e-3              \\
        --eval-games 50        \\
        --eval-every 5         \\
        --checkpoint-every 10  \\
        --checkpoint-dir models \\
        --max-moves 60         \\
        --device cpu
"""

import argparse
import logging
import sys
import os
import time

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so `src.*` imports work when
# the script is invoked directly from the repo root.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.models.policy_network import PolicyNetwork
from src.training.self_play import run_games, records_to_dataset
from src.training.train_policy import (
    TrainingConfig,
    compute_loss,
    save_checkpoint,
    load_checkpoint,
    EpochMetrics,
)
from src.evaluation.evaluate_model import evaluate, EvaluationConfig
from src.training.pgn_writer import records_to_pgn, save_pgn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Logger — writes to both console and logs/train.log simultaneously
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str) -> logging.Logger:
    """
    Configure a logger that mirrors all output to both the terminal and a
    timestamped log file inside `log_dir`.

    File format  : logs/train_YYYYMMDD_HHMMSS.log
    Console output: plain message (no timestamp clutter on screen)
    File output  : timestamp + level + message (full context for debugging)

    Returns
    -------
    logging.Logger  — use log.info() in place of print() throughout main().
    """
    os.makedirs(log_dir, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(log_dir, f"train_{timestamp}.log")

    logger = logging.getLogger("chess_rl")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()   # avoid duplicate handlers on re-runs

    # File handler — full detail
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # Console handler — clean, no timestamp
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path}")
    return logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the Chess RL policy network via self-play.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training duration
    p.add_argument("--epochs",           type=int,   default=50,
                   help="Number of training epochs.")
    p.add_argument("--games-per-epoch",  type=int,   default=20,
                   help="Self-play games to generate each epoch.")

    # Optimiser
    p.add_argument("--lr",               type=float, default=1e-3,
                   help="Adam learning rate.")

    # Self-play
    p.add_argument("--max-moves",        type=int,   default=60,
                   help="Move cap per self-play game (capped games = draw).")
    p.add_argument("--temp-high",        type=float, default=1.0,
                   help="Sampling temperature for early-game moves.")
    p.add_argument("--temp-low",         type=float, default=0.1,
                   help="Sampling temperature for late-game moves.")
    p.add_argument("--temp-threshold",   type=int,   default=30,
                   help="Half-move number at which temperature drops.")

    # Checkpointing
    p.add_argument("--checkpoint-dir",   type=str,   default="models",
                   help="Directory to save model checkpoints.")
    p.add_argument("--checkpoint-every", type=int,   default=10,
                   help="Save a checkpoint every N epochs (0 = never).")
    p.add_argument("--resume",           type=str,   default=None,
                   help="Path to a checkpoint .pt file to resume from.")

    # Evaluation
    p.add_argument("--eval-games",       type=int,   default=50,
                   help="Games to play against the random agent per evaluation.")
    p.add_argument("--eval-every",       type=int,   default=5,
                   help="Run evaluation every N epochs (0 = never, -1 = end only).")

    # Stockfish
    p.add_argument("--no-stockfish",     action="store_true",
                   help="Disable Stockfish evaluation at move cap (cap = draw).")

    # Hardware
    p.add_argument("--device",           type=str,   default="cpu",
                   help="Torch device ('cpu' or 'cuda').")
    p.add_argument("--log-dir",          type=str,   default="logs",
                   help="Directory to write training log files.")
    p.add_argument("--pgn-dir",          type=str,   default=None,
                   help="If set, save self-play games as PGN to this directory every epoch.")
    p.add_argument("--pgn-every",        type=int,   default=10,
                   help="Save PGN every N epochs (only used when --pgn-dir is set).")

    return p


# ---------------------------------------------------------------------------
# Epoch summary printer
# ---------------------------------------------------------------------------

def print_epoch_header(epoch: int, total: int, log: logging.Logger) -> None:
    log.info(f"\n{'─'*60}")
    log.info(f"  Epoch {epoch}/{total}")
    log.info(f"{'─'*60}")


def print_self_play_summary(records: list, samples: list, log: logging.Logger) -> None:
    wins      = sum(1 for r in records if r.result == "white_wins")
    losses    = sum(1 for r in records if r.result == "black_wins")
    draws     = sum(1 for r in records if r.result == "draw")
    cap_dec   = sum(1 for r in records if r.result == "max_moves_reached")   # cap + decisive SF
    cap_draw  = sum(1 for r in records if r.result == "max_moves_draw")       # cap + equal SF
    sf_used   = sum(1 for r in records if r.stockfish_cp is not None)
    avg_len   = sum(r.n_moves for r in records) / len(records) if records else 0.0
    log.info(f"  Self-play games   : {len(records)}")
    log.info(f"  Samples collected : {len(samples)}")
    log.info(f"  Outcomes          : W {wins} / L {losses} / D {draws} / Cap-decisive {cap_dec} / Cap-draw {cap_draw}")
    log.info(f"  Stockfish evals   : {sf_used}")
    log.info(f"  Avg game length   : {avg_len:.1f} half-moves")


def print_training_summary(loss: torch.Tensor, updated: bool, log: logging.Logger) -> None:
    if updated:
        log.info(f"  Training loss     : {loss.item():.6f}")
    else:
        log.info(f"  Training loss     : 0.000000  (all draws — no update)")


def print_eval_summary(result, log: logging.Logger) -> None:
    log.info(
        f"  Evaluation        : "
        f"W {result.wins} / L {result.losses} / D {result.draws + result.max_moves_games}"
        f"  →  Winrate {result.winrate * 100:.1f}%"
    )


# ---------------------------------------------------------------------------
# Core training step (one epoch)
# ---------------------------------------------------------------------------

def run_epoch(
    model      : PolicyNetwork,
    optimiser  : torch.optim.Optimizer,
    config     : argparse.Namespace,
    epoch      : int,
) -> tuple[torch.Tensor, list, list]:
    """
    Run one full epoch: self-play → loss → update.

    Returns
    -------
    loss    : torch.Tensor  Scalar loss (may have requires_grad=False if all draws)
    records : list[GameRecord]
    samples : list[GameSample]
    """
    # ── Self-play ──────────────────────────────────────────────────────
    model.eval()
    records = run_games(
        model,
        n_games        = config.games_per_epoch,
        max_moves      = config.max_moves,
        temp_high      = config.temp_high,
        temp_low       = config.temp_low,
        temp_threshold = config.temp_threshold,
        device         = config.device,
        use_stockfish  = not getattr(config, "no_stockfish", False),
    )
    samples = records_to_dataset(records)

    # ── Training step ──────────────────────────────────────────────────
    model.train()
    optimiser.zero_grad()
    loss    = compute_loss(model, samples, device=config.device)
    updated = loss.requires_grad

    if updated:
        loss.backward()
        optimiser.step()

    return loss, records, samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # ── Logging ───────────────────────────────────────────────────
    log = setup_logger(args.log_dir)

    # ── Banner ────────────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info(f"  Chess RL Agent — Training")
    log.info(f"{'='*60}")
    log.info(f"  Epochs          : {args.epochs}")
    log.info(f"  Games / epoch   : {args.games_per_epoch}")
    log.info(f"  Learning rate   : {args.lr}")
    log.info(f"  Max moves       : {args.max_moves}")
    log.info(f"  Checkpoint dir  : {args.checkpoint_dir}")
    log.info(f"  Checkpoint every: {args.checkpoint_every} epochs")
    log.info(f"  Eval every      : {args.eval_every} epochs")
    log.info(f"  Eval games      : {args.eval_games}")
    log.info(f"  Device          : {args.device}")
    log.info(f"  Stockfish       : {'disabled (--no-stockfish)' if args.no_stockfish else 'enabled (at move cap)'}")
    log.info(f"  PGN dir         : {args.pgn_dir or '(disabled)'}")
    log.info(f"{'='*60}")

    # ── Model ─────────────────────────────────────────────────────────
    model = PolicyNetwork().to(args.device)
    log.info(f"\n  Parameters      : {model.count_parameters():,}")

    start_epoch = 1
    if args.resume:
        if not os.path.exists(args.resume):
            log.info(f"\n  ERROR: checkpoint not found: {args.resume}")
            sys.exit(1)
        start_epoch = load_checkpoint(args.resume, model) + 1
        log.info(f"  Resumed from    : {args.resume}  (epoch {start_epoch - 1})")

    # ── Optimiser ─────────────────────────────────────────────────────
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Training loop ─────────────────────────────────────────────────
    history: list[EpochMetrics] = []
    train_start = time.time()

    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_start = time.time()
        print_epoch_header(epoch, start_epoch + args.epochs - 1, log)

        loss, records, samples = run_epoch(model, optimiser, args, epoch)
        updated = loss.requires_grad  # False when all games were draws

        print_self_play_summary(records, samples, log)
        print_training_summary(loss, updated, log)

        # ── Checkpoint ────────────────────────────────────────────────
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            # Build a minimal EpochMetrics for the checkpoint metadata
            wins   = sum(1 for r in records if r.result == "white_wins")
            losses = sum(1 for r in records if r.result == "black_wins")
            draws  = sum(1 for r in records if r.result in ("draw", "max_moves_reached", "max_moves_draw"))
            m = EpochMetrics(
                epoch=epoch, loss=loss.item(),
                n_samples=len(samples), n_games=len(records),
                wins=wins, losses=losses, draws=draws,
                max_moves_hit=sum(1 for r in records if r.result == "max_moves_reached"),
                duration_sec=time.time() - epoch_start,
            )
            path = save_checkpoint(model, epoch, m, args.checkpoint_dir)
            log.info(f"  Checkpoint saved  : {path}")

        # ── PGN saving ────────────────────────────────────────────────
        if args.pgn_dir and args.pgn_every > 0 and epoch % args.pgn_every == 0:
            pgn_text = records_to_pgn(records, epoch=epoch)
            pgn_path = save_pgn(pgn_text, epoch=epoch, pgn_dir=args.pgn_dir)
            log.info(f"  PGN saved         : {pgn_path}")

        # ── Evaluation ────────────────────────────────────────────────
        run_eval = False
        if args.eval_every > 0 and epoch % args.eval_every == 0:
            run_eval = True
        elif args.eval_every == -1 and epoch == (start_epoch + args.epochs - 1):
            run_eval = True

        if run_eval and args.eval_games > 0:
            eval_cfg = EvaluationConfig(
                n_games    = args.eval_games,
                max_moves  = args.max_moves,
                temperature= args.temp_low,
                device     = args.device,
            )
            eval_result = evaluate(model, eval_cfg)
            print_eval_summary(eval_result, log)

        log.info(f"  Epoch time        : {time.time() - epoch_start:.1f}s")

    # ── Final summary ────────────────────────────────────────────────
    total_time = time.time() - train_start
    log.info(f"\n{'='*60}")
    log.info(f"  Training complete")
    log.info(f"  Total time      : {total_time:.1f}s  ({total_time/60:.1f} min)")
    log.info(f"  Epochs trained  : {args.epochs}")
    log.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()