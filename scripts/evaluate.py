"""
scripts/evaluate.py
-------------------
Standalone evaluation script for the Chess RL Agent.

Loads a saved model checkpoint and evaluates it against the random agent,
printing a full result summary to the console.

Usage
-----
    # Evaluate a specific checkpoint
    python scripts/evaluate.py --model models/policy_epoch_0050.pt

    # Use more games for a more reliable estimate
    python scripts/evaluate.py --model models/policy_epoch_0050.pt --games 200

    # Save results to a JSON file
    python scripts/evaluate.py --model models/policy_epoch_0050.pt \\
        --save-path data/eval/epoch_50_results.json

    # Show per-game output
    python scripts/evaluate.py --model models/policy_epoch_0050.pt --verbose

    # Compare multiple checkpoints quickly
    for epoch in 10 20 30 40 50; do
        python scripts/evaluate.py --model models/policy_epoch_00${epoch}.pt
    done
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.policy_network import PolicyNetwork
from src.training.train_policy import load_checkpoint
from src.evaluation.evaluate_model import evaluate, EvaluationConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a Chess RL model checkpoint against the random agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", type=str, required=True,
        help="Path to a model checkpoint .pt file.",
    )
    p.add_argument(
        "--games", type=int, default=100,
        help="Number of games to play against the random agent.",
    )
    p.add_argument(
        "--max-moves", type=int, default=200,
        help="Move cap per game — capped games count as draws.",
    )
    p.add_argument(
        "--temperature", type=float, default=0.1,
        help="Model sampling temperature (0.1 = near-greedy, 1.0 = exploratory).",
    )
    p.add_argument(
        "--no-alternate", action="store_true",
        help="If set, model always plays White (default: alternates sides).",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for the random agent (for reproducibility).",
    )
    p.add_argument(
        "--save-path", type=str, default=None,
        help="If set, save evaluation results as JSON to this path.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print a one-line summary for every individual game.",
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device ('cpu' or 'cuda').",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # ── Load checkpoint ──────────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"\n  ERROR: checkpoint not found: {args.model}")
        sys.exit(1)

    model = PolicyNetwork()
    epoch = load_checkpoint(args.model, model)
    model.to(args.device)
    model.eval()

    print(f"\n{'='*60}")
    print(f"  Chess RL Agent — Evaluation")
    print(f"{'='*60}")
    print(f"  Checkpoint      : {args.model}")
    print(f"  Trained epochs  : {epoch}")
    print(f"  Parameters      : {model.count_parameters():,}")
    print(f"  Games           : {args.games}")
    print(f"  Temperature     : {args.temperature}")
    print(f"  Alternating     : {not args.no_alternate}")
    print(f"  Device          : {args.device}")
    print(f"{'='*60}")

    # ── Run evaluation ───────────────────────────────────────────────
    cfg = EvaluationConfig(
        n_games         = args.games,
        max_moves       = args.max_moves,
        temperature     = args.temperature,
        random_seed     = args.seed,
        alternate_sides = not args.no_alternate,
        save_path       = args.save_path,
        verbose         = args.verbose,
        device          = args.device,
    )

    result = evaluate(model, cfg)

    # ── Exit code reflects performance ───────────────────────────────
    # Exit 0 if winrate > 50%, exit 1 otherwise.
    # Useful for CI pipelines or shell scripts that check model quality.
    sys.exit(0 if result.winrate >= 0.5 else 1)


if __name__ == "__main__":
    main()