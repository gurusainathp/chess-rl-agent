"""
scripts/train.py
----------------
Main training entry point for the Chess RL Agent.

Thin CLI wrapper around `train_policy.train()`.  All logic lives there —
this script just parses arguments, builds a TrainingConfig, and calls train().

Phases implemented
------------------
  1  Temperature sampling       --temp-high / --temp-low / --temp-threshold
  2  Entropy regularisation     --entropy-coeff
  3  Opponent pool              --pool-self / --pool-checkpoint / ...
  4  Random agent               (always active in pool)
  5  Checkpoint agent           (auto-loaded after first checkpoint save)
  6  Replay buffer              --replay-capacity / --replay-batch-size
  7  Opening randomisation      --opening-moves / --opening-prob
  8  Depth-limited Stockfish    --stockfish-depth (pool + move-cap eval)
  9  Extended evaluation        --eval-vs-checkpoint-games / --eval-vs-stockfish-games

Usage
-----
    # All defaults (50 epochs, 20 games/epoch)
    python scripts/train.py

    # Recommended overnight run with all phases active
    python scripts/train.py \\
        --epochs 300                      \\
        --games-per-epoch 60              \\
        --max-moves 60                    \\
        --replay-capacity 100000          \\
        --replay-batch-size 2048          \\
        --opening-moves 4                 \\
        --eval-every 10                   \\
        --eval-vs-checkpoint-games 50     \\
        --eval-vs-stockfish-games 20      \\
        --eval-log-path logs/eval_results.json \\
        --checkpoint-dir models           \\
        --log-dir logs

    # Resume from checkpoint
    python scripts/train.py --resume models/policy_epoch_0050.pt --epochs 100
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.policy_network import PolicyNetwork
from src.training.train_policy import TrainingConfig, train, load_checkpoint


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the Chess RL policy network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Training duration ──────────────────────────────────────────────────
    g = p.add_argument_group("Training duration")
    g.add_argument("--epochs",              type=int,   default=50,
                   help="Number of training epochs.")
    g.add_argument("--games-per-epoch",     type=int,   default=20,
                   help="Games generated per epoch (self-play + pool).")

    # ── Optimiser ──────────────────────────────────────────────────────────
    g = p.add_argument_group("Optimiser")
    g.add_argument("--lr",                  type=float, default=1e-3,
                   help="Adam learning rate.")
    g.add_argument("--entropy-coeff",       type=float, default=0.01,
                   help="Entropy regularisation coefficient (Phase 2).")

    # ── Self-play / temperature (Phase 1) ──────────────────────────────────
    g = p.add_argument_group("Self-play / Temperature (Phase 1)")
    g.add_argument("--max-moves",           type=int,   default=60,
                   help="Move cap per game; capped games scored by Stockfish.")
    g.add_argument("--temp-high",           type=float, default=1.0,
                   help="Sampling temperature for early-game moves.")
    g.add_argument("--temp-low",            type=float, default=0.1,
                   help="Sampling temperature for late-game moves.")
    g.add_argument("--temp-threshold",      type=int,   default=30,
                   help="Half-move at which temperature drops to --temp-low.")

    # ── Opening randomisation (Phase 7) ────────────────────────────────────
    g = p.add_argument_group("Opening randomisation (Phase 7)")
    g.add_argument("--opening-moves",       type=int,   default=4,
                   help="Random half-moves played before model takes control. "
                        "0 = always start from standard position.")
    g.add_argument("--opening-prob",        type=float, default=1.0,
                   help="Fraction of games that apply opening randomisation.")

    # ── Replay buffer (Phase 6) ────────────────────────────────────────────
    g = p.add_argument_group("Replay buffer (Phase 6)")
    g.add_argument("--no-replay",           action="store_true",
                   help="Disable replay buffer (train only on current-epoch games).")
    g.add_argument("--replay-capacity",     type=int,   default=200_000,
                   help="Max positions stored in the replay buffer.")
    g.add_argument("--replay-batch-size",   type=int,   default=2_048,
                   help="Positions sampled from the buffer per training step.")

    # ── Opponent pool (Phase 3-5) ──────────────────────────────────────────
    g = p.add_argument_group("Opponent pool (Phases 3-5)")
    g.add_argument("--no-opponent-pool",    action="store_true",
                   help="Disable opponent pool (pure self-play only).")
    g.add_argument("--pool-size",           type=int,   default=5,
                   help="Max past checkpoints kept as pool opponents.")
    g.add_argument("--pool-self",           type=float, default=0.40,
                   help="Sampling weight for self-play games (max 40%%).")
    g.add_argument("--pool-stockfish",      type=float, default=0.30,
                   help="Sampling weight for Stockfish. "
                        "Checkpoint folds here if unavailable; "
                        "then folds to random if Stockfish also unavailable.")
    g.add_argument("--pool-checkpoint",     type=float, default=0.20,
                   help="Sampling weight for games vs past checkpoints.")
    g.add_argument("--pool-random",         type=float, default=0.10,
                   help="Sampling weight for games vs random agent.")

    # ── Stockfish (Phase 8) ────────────────────────────────────────────────
    g = p.add_argument_group("Stockfish (Phase 8)")
    g.add_argument("--no-stockfish",        action="store_true",
                   help="Disable Stockfish entirely. Move-cap = draw; "
                        "Stockfish pool slot folds into random.")
    g.add_argument("--stockfish-depth",     type=int,   default=8,
                   help="UCI search depth for Stockfish as pool opponent "
                        "and move-cap evaluator (5 ~ 1600 ELO).")

    # ── Checkpointing ──────────────────────────────────────────────────────
    g = p.add_argument_group("Checkpointing")
    g.add_argument("--checkpoint-dir",      type=str,   default="models",
                   help="Directory to save model checkpoints.")
    g.add_argument("--checkpoint-every",    type=int,   default=10,
                   help="Save a checkpoint every N epochs (0 = never).")
    g.add_argument("--resume",              type=str,   default=None,
                   help="Path to a .pt checkpoint file to resume training.")

    # ── Evaluation (Phase 9) ───────────────────────────────────────────────
    g = p.add_argument_group("Evaluation (Phase 9)")
    g.add_argument("--eval-every",                  type=int,   default=10,
                   help="Run evaluation every N epochs. 0=never, -1=end only.")
    g.add_argument("--eval-games",                  type=int,   default=100,
                   help="Tier 1: games vs RandomAgent per evaluation run.")
    g.add_argument("--eval-vs-checkpoint-games",    type=int,   default=50,
                   help="Tier 2: games vs most-recent checkpoint (0 = skip).")
    g.add_argument("--eval-vs-stockfish-games",     type=int,   default=20,
                   help="Tier 3: games vs depth-limited Stockfish (0 = skip).")
    g.add_argument("--eval-log-path",               type=str,
                   default="logs/eval_results.json",
                   help="JSON file to append all evaluation results to.")

    # ── PGN ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("PGN saving")
    g.add_argument("--pgn-dir",             type=str,   default=None,
                   help="Save self-play games as PGN to this directory.")
    g.add_argument("--pgn-every",           type=int,   default=10,
                   help="Save PGN every N epochs (requires --pgn-dir).")

    # ── Hardware / misc ────────────────────────────────────────────────────
    g = p.add_argument_group("Hardware / misc")
    g.add_argument("--device",              type=str,   default="cpu",
                   help="Torch device: 'cpu' or 'cuda'.")
    g.add_argument("--log-dir",             type=str,   default="logs",
                   help="Directory for timestamped training .log files.")
    g.add_argument("--verbose",             action="store_true",
                   help="Print per-game output during self-play.")

    return p


def main() -> None:
    args = build_parser().parse_args()

    config = TrainingConfig(
        # Duration
        n_epochs                  = args.epochs,
        games_per_epoch           = args.games_per_epoch,
        # Optimiser
        learning_rate             = args.lr,
        entropy_coeff             = args.entropy_coeff,
        # Self-play / temperature (Phase 1)
        max_moves                 = args.max_moves,
        temp_high                 = args.temp_high,
        temp_low                  = args.temp_low,
        temp_threshold            = args.temp_threshold,
        # Opening randomisation (Phase 7)
        opening_moves             = args.opening_moves,
        opening_prob              = args.opening_prob,
        # Replay buffer (Phase 6)
        use_replay_buffer         = not args.no_replay,
        replay_capacity           = args.replay_capacity,
        replay_batch_size         = args.replay_batch_size,
        # Opponent pool (Phases 3-5)
        use_opponent_pool         = not args.no_opponent_pool,
        pool_size                 = args.pool_size,
        pool_weight_self          = args.pool_self,
        pool_weight_checkpoint    = args.pool_checkpoint,
        pool_weight_random        = args.pool_random,
        pool_weight_stockfish     = args.pool_stockfish,
        # Stockfish (Phase 8)
        use_stockfish             = not args.no_stockfish,
        eval_stockfish_depth      = args.stockfish_depth,
        # Checkpointing
        checkpoint_dir            = args.checkpoint_dir,
        checkpoint_every          = args.checkpoint_every,
        # Evaluation (Phase 9)
        eval_every                = args.eval_every,
        eval_games                = args.eval_games,
        eval_vs_checkpoint_games  = args.eval_vs_checkpoint_games,
        eval_vs_stockfish_games   = args.eval_vs_stockfish_games,
        eval_log_path             = args.eval_log_path,
        # PGN
        pgn_dir                   = args.pgn_dir,
        pgn_every                 = args.pgn_every,
        # Hardware / misc
        device                    = args.device,
        log_dir                   = args.log_dir,
        verbose                   = args.verbose,
    )

    # Load weights before handing off; train() calls model.to(device) itself.
    model = PolicyNetwork()
    start_epoch = 0
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"ERROR: checkpoint not found: {args.resume}")
            sys.exit(1)
        start_epoch = load_checkpoint(args.resume, model)
        print(f"Resumed from: {args.resume}  (epoch {start_epoch})")

    train(config=config, model=model, start_epoch=start_epoch)


if __name__ == "__main__":
    main()