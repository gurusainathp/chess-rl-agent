"""
scripts/train.py
----------------
Main training entry point for the Chess RL Agent.

Thin CLI wrapper around `train_policy.train()`.  All logic lives there —
this script just parses arguments, builds a TrainingConfig, and calls train().

Usage
-----
    # Run with all defaults
    python scripts/train.py

    # Common overrides
    python scripts/train.py --epochs 100 --games-per-epoch 30 --lr 5e-4

    # Resume from a checkpoint
    python scripts/train.py --resume models/policy_epoch_0010.pt

    # Full recommended overnight run
    python scripts/train.py \\
        --epochs 300           \\
        --games-per-epoch 60   \\
        --lr 1e-3              \\
        --eval-games 50        \\
        --eval-every 10        \\
        --checkpoint-every 10  \\
        --checkpoint-dir models \\
        --max-moves 60         \\
        --device cpu
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.models.policy_network import PolicyNetwork
from src.training.train_policy import TrainingConfig, train, load_checkpoint


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the Chess RL policy network via self-play.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training duration
    p.add_argument("--epochs",           type=int,   default=50,
                   help="Number of training epochs.")
    p.add_argument("--games-per-epoch",  type=int,   default=20,
                   help="Self-play / pool games per epoch.")

    # Optimiser
    p.add_argument("--lr",               type=float, default=1e-3,
                   help="Adam learning rate.")

    # Loss
    p.add_argument("--entropy-coeff",    type=float, default=0.01,
                   help="Entropy regularisation coefficient.")

    # Self-play
    p.add_argument("--max-moves",        type=int,   default=60,
                   help="Move cap per game (capped games evaluated by Stockfish).")
    p.add_argument("--temp-high",        type=float, default=1.0,
                   help="Sampling temperature for early-game moves.")
    p.add_argument("--temp-low",         type=float, default=0.1,
                   help="Sampling temperature for late-game moves.")
    p.add_argument("--temp-threshold",   type=int,   default=30,
                   help="Half-move at which temperature drops to temp-low.")

    # Opponent pool
    p.add_argument("--no-opponent-pool", action="store_true",
                   help="Disable opponent pool (pure self-play only).")
    p.add_argument("--pool-size",        type=int,   default=5,
                   help="Max checkpoint opponents kept in pool.")
    p.add_argument("--pool-self",        type=float, default=0.60,
                   help="Fraction of games played as self-play.")
    p.add_argument("--pool-checkpoint",  type=float, default=0.20,
                   help="Fraction of games vs past checkpoints.")
    p.add_argument("--pool-random",      type=float, default=0.10,
                   help="Fraction of games vs random agent.")
    p.add_argument("--pool-stockfish",   type=float, default=0.10,
                   help="Fraction of games vs Stockfish (falls back to random if unavailable).")

    # Checkpointing
    p.add_argument("--checkpoint-dir",   type=str,   default="models",
                   help="Directory to save model checkpoints.")
    p.add_argument("--checkpoint-every", type=int,   default=10,
                   help="Save a checkpoint every N epochs (0 = never).")
    p.add_argument("--resume",           type=str,   default=None,
                   help="Path to a .pt checkpoint to resume from.")

    # Evaluation
    p.add_argument("--eval-games",       type=int,   default=50,
                   help="Games vs random agent per evaluation run.")
    p.add_argument("--eval-every",       type=int,   default=10,
                   help="Run evaluation every N epochs (0 = never, -1 = end only).")

    # Stockfish
    p.add_argument("--no-stockfish",     action="store_true",
                   help="Disable Stockfish evaluation at move cap (cap = draw).")

    # PGN
    p.add_argument("--pgn-dir",          type=str,   default=None,
                   help="If set, save self-play PGN to this directory.")
    p.add_argument("--pgn-every",        type=int,   default=10,
                   help="Save PGN every N epochs (only when --pgn-dir is set).")

    # Hardware / misc
    p.add_argument("--device",           type=str,   default="cpu",
                   help="Torch device ('cpu' or 'cuda').")
    p.add_argument("--log-dir",          type=str,   default="logs",
                   help="Directory for timestamped training log files.")
    p.add_argument("--verbose",          action="store_true",
                   help="Print per-game output during self-play.")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    # Build config — single source of truth passed to train()
    config = TrainingConfig(
        n_epochs              = args.epochs,
        games_per_epoch       = args.games_per_epoch,
        learning_rate         = args.lr,
        entropy_coeff         = args.entropy_coeff,
        max_moves             = args.max_moves,
        temp_high             = args.temp_high,
        temp_low              = args.temp_low,
        temp_threshold        = args.temp_threshold,
        use_opponent_pool     = not args.no_opponent_pool,
        pool_size             = args.pool_size,
        pool_weight_self      = args.pool_self,
        pool_weight_checkpoint= args.pool_checkpoint,
        pool_weight_random    = args.pool_random,
        pool_weight_stockfish = args.pool_stockfish,
        checkpoint_dir        = args.checkpoint_dir,
        checkpoint_every      = args.checkpoint_every,
        use_stockfish         = not args.no_stockfish,
        eval_games            = args.eval_games,
        eval_every            = args.eval_every,
        pgn_dir               = args.pgn_dir,
        pgn_every             = args.pgn_every,
        log_dir               = args.log_dir,
        device                = args.device,
        verbose               = args.verbose,
    )

    # Optionally resume from checkpoint.
    # Load weights first (always to CPU via map_location), THEN move to device.
    # This avoids a redundant .to(device) call that train() would do again.
    model = PolicyNetwork()
    start_epoch = 0
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"ERROR: checkpoint not found: {args.resume}")
            sys.exit(1)
        start_epoch = load_checkpoint(args.resume, model)   # loads to CPU
        print(f"Resumed from: {args.resume}  (epoch {start_epoch})")
    # train() calls model.to(config.device) internally — don't do it here

    # Hand off — all training logic lives in train_policy.train()
    train(config=config, model=model, start_epoch=start_epoch)


if __name__ == "__main__":
    main()