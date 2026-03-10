"""
scripts/compare_checkpoints.py
-------------------------------
Head-to-head evaluation between two model checkpoints.

Plays N games between two saved checkpoints and reports the result from
the perspective of the CHALLENGER (the newer / stronger candidate model).

This answers the key question:
    "Is epoch 80 actually stronger than epoch 40?"

The two models alternate sides every game (same bias-removal logic as
evaluate_model.py) so neither colour advantage inflates the result.

Usage
-----
    # Compare two checkpoints
    python scripts/compare_checkpoints.py \\
        --model-a models/policy_epoch_0080.pt \\
        --model-b models/policy_epoch_0040.pt

    # More games for a reliable estimate
    python scripts/compare_checkpoints.py \\
        --model-a models/policy_epoch_0080.pt \\
        --model-b models/policy_epoch_0040.pt \\
        --games 200

    # Auto-compare every checkpoint in a directory (latest vs each previous)
    python scripts/compare_checkpoints.py --dir models/ --games 50

    # Save result to JSON
    python scripts/compare_checkpoints.py \\
        --model-a models/policy_epoch_0080.pt \\
        --model-b models/policy_epoch_0040.pt \\
        --save-path results/comparison.json
"""

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chess
import torch

from src.environment.board_encoder import encode_board
from src.models.policy_network import PolicyNetwork
from src.training.train_policy import load_checkpoint


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """
    Result of a head-to-head match between two model checkpoints.

    All outcomes are from Model A's perspective.

    Attributes
    ----------
    model_a_path  : str   Path to checkpoint A (challenger).
    model_b_path  : str   Path to checkpoint B (baseline).
    model_a_epoch : int   Epoch of checkpoint A.
    model_b_epoch : int   Epoch of checkpoint B.
    total_games   : int
    a_wins        : int   Games won by Model A.
    b_wins        : int   Games won by Model B.
    draws         : int   Draws and capped games combined.
    a_winrate     : float Fraction of games won by A  [0.0–1.0].
    avg_length    : float Average half-moves per game.
    duration_sec  : float Wall-clock seconds for the match.
    """
    model_a_path  : str
    model_b_path  : str
    model_a_epoch : int
    model_b_epoch : int
    total_games   : int
    a_wins        : int
    b_wins        : int
    draws         : int
    a_winrate     : float
    avg_length    : float
    duration_sec  : float

    def summary(self) -> str:
        bar = "=" * 60
        a_label = f"Model A  (epoch {self.model_a_epoch})"
        b_label = f"Model B  (epoch {self.model_b_epoch})"
        verdict = (
            "Model A is STRONGER ✓"  if self.a_winrate > 0.5  else
            "Model B is STRONGER"    if self.a_winrate < 0.5  else
            "DRAW — roughly equal"
        )
        return (
            f"\n{bar}\n"
            f"  Head-to-Head: {os.path.basename(self.model_a_path)}\n"
            f"           vs  {os.path.basename(self.model_b_path)}\n"
            f"{bar}\n"
            f"  {a_label:<35} {self.a_wins:>4} wins  ({self.a_winrate*100:.1f}%)\n"
            f"  {b_label:<35} {self.b_wins:>4} wins  ({(1-self.a_winrate)*100:.1f}%)\n"
            f"  Draws / Capped         : {self.draws:>4}\n"
            f"  Total games            : {self.total_games}\n"
            f"  Avg game length        : {self.avg_length:.1f} half-moves\n"
            f"  Duration               : {self.duration_sec:.1f}s\n"
            f"\n  Verdict  →  {verdict}\n"
            f"{bar}\n"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Single-game engine
# ---------------------------------------------------------------------------

def _select_move(
    model      : PolicyNetwork,
    board      : chess.Board,
    temperature: float,
    device     : str,
) -> chess.Move:
    """Run the policy network to select one move."""
    legal   = list(board.legal_moves)
    state_t = torch.tensor(
        encode_board(board), dtype=torch.float32, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        return model.select_move(state_t, legal, temperature=temperature)


def _play_one_game(
    model_white : PolicyNetwork,
    model_black : PolicyNetwork,
    max_moves   : int,
    temperature : float,
    device      : str,
) -> tuple[str, int]:
    """
    Play one game between two models.

    model_white plays White, model_black plays Black.

    Returns
    -------
    result  : str   'white', 'black', or 'draw'
    n_moves : int   Number of half-moves played.
    """
    board   = chess.Board()
    n_moves = 0

    for _ in range(max_moves):
        if board.is_game_over():
            break
        if not list(board.legal_moves):
            break

        model = model_white if board.turn == chess.WHITE else model_black
        move  = _select_move(model, board, temperature, device)
        board.push(move)
        n_moves += 1

    if not board.is_game_over():
        return "draw", n_moves

    res = board.result()
    if res == "1-0":
        return "white", n_moves
    elif res == "0-1":
        return "black", n_moves
    else:
        return "draw", n_moves


# ---------------------------------------------------------------------------
# Head-to-head match
# ---------------------------------------------------------------------------

def run_match(
    model_a     : PolicyNetwork,
    model_b     : PolicyNetwork,
    model_a_path: str,
    model_b_path: str,
    model_a_epoch: int,
    model_b_epoch: int,
    n_games     : int   = 100,
    max_moves   : int   = 200,
    temperature : float = 0.1,
    device      : str   = "cpu",
    verbose     : bool  = False,
) -> MatchResult:
    """
    Play n_games between model_a and model_b.

    Models alternate colours every game to remove first-move bias.
    Even-numbered games: A plays White.
    Odd-numbered  games: B plays White.

    Outcomes are always reported from Model A's perspective.

    Parameters
    ----------
    model_a, model_b : PolicyNetwork
    n_games          : int
    max_moves        : int
    temperature      : float   Low = near-greedy (recommended for evaluation).
    device           : str
    verbose          : bool    Print per-game output.

    Returns
    -------
    MatchResult
    """
    model_a.eval()
    model_b.eval()

    a_wins = b_wins = draws = total_moves = 0
    start  = time.time()

    for i in range(n_games):
        a_is_white = (i % 2 == 0)
        mw = model_a if a_is_white else model_b
        mb = model_b if a_is_white else model_a

        raw_result, n_moves = _play_one_game(mw, mb, max_moves, temperature, device)
        total_moves += n_moves

        # Map raw result to A/B perspective
        if raw_result == "draw":
            draws += 1
            outcome = "draw"
        elif (raw_result == "white" and a_is_white) or \
             (raw_result == "black" and not a_is_white):
            a_wins += 1
            outcome = "A wins"
        else:
            b_wins += 1
            outcome = "B wins"

        if verbose:
            sides = f"A={'W' if a_is_white else 'B'}"
            print(
                f"  Game {i+1:>4}/{n_games} | {sides} | "
                f"Outcome: {outcome:<8} | Moves: {n_moves}"
            )

    elapsed  = time.time() - start
    a_winrate = a_wins / n_games if n_games > 0 else 0.0
    avg_len   = total_moves / n_games if n_games > 0 else 0.0

    return MatchResult(
        model_a_path=model_a_path,
        model_b_path=model_b_path,
        model_a_epoch=model_a_epoch,
        model_b_epoch=model_b_epoch,
        total_games=n_games,
        a_wins=a_wins,
        b_wins=b_wins,
        draws=draws,
        a_winrate=a_winrate,
        avg_length=avg_len,
        duration_sec=elapsed,
    )


# ---------------------------------------------------------------------------
# Directory mode — compare all checkpoints against the latest
# ---------------------------------------------------------------------------

def run_directory_comparison(
    checkpoint_dir: str,
    n_games       : int,
    max_moves     : int,
    temperature   : float,
    device        : str,
    save_dir      : str | None,
) -> None:
    """
    Auto-compare every checkpoint in a directory against the most recent one.

    Checkpoints are sorted by epoch number (parsed from filename).
    The latest checkpoint is Model A (challenger); all others are Model B.
    """
    pattern = os.path.join(checkpoint_dir, "policy_epoch_*.pt")
    files   = sorted(glob.glob(pattern))

    if len(files) < 2:
        print(f"  Need at least 2 checkpoints in '{checkpoint_dir}', found {len(files)}.")
        sys.exit(1)

    latest_path  = files[-1]
    latest_model = PolicyNetwork()
    latest_epoch = load_checkpoint(latest_path, latest_model)
    latest_model.to(device)

    print(f"\n  Challenger (Model A): {os.path.basename(latest_path)}  (epoch {latest_epoch})")
    print(f"  Comparing against {len(files)-1} earlier checkpoint(s).\n")

    for path in files[:-1]:
        baseline       = PolicyNetwork()
        baseline_epoch = load_checkpoint(path, baseline)
        baseline.to(device)

        print(f"  vs {os.path.basename(path)}  (epoch {baseline_epoch}) ...")
        result = run_match(
            model_a=latest_model, model_b=baseline,
            model_a_path=latest_path, model_b_path=path,
            model_a_epoch=latest_epoch, model_b_epoch=baseline_epoch,
            n_games=n_games, max_moves=max_moves,
            temperature=temperature, device=device,
        )
        print(result.summary())

        if save_dir:
            fname = f"match_{latest_epoch}_vs_{baseline_epoch}.json"
            result.save_json(os.path.join(save_dir, fname))
            print(f"  Saved: {os.path.join(save_dir, fname)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Head-to-head evaluation between two Chess RL checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Two-checkpoint mode
    p.add_argument("--model-a", type=str, default=None,
                   help="Path to checkpoint A (challenger / newer model).")
    p.add_argument("--model-b", type=str, default=None,
                   help="Path to checkpoint B (baseline / older model).")

    # Directory mode
    p.add_argument("--dir", type=str, default=None,
                   help="Compare all checkpoints in this directory against the latest.")

    # Shared settings
    p.add_argument("--games",       type=int,   default=100,
                   help="Number of games per match.")
    p.add_argument("--max-moves",   type=int,   default=200,
                   help="Move cap per game.")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="Sampling temperature for both models.")
    p.add_argument("--device",      type=str,   default="cpu",
                   help="Torch device.")
    p.add_argument("--verbose",     action="store_true",
                   help="Print per-game output.")
    p.add_argument("--save-path",   type=str,   default=None,
                   help="Save result JSON to this path (two-checkpoint mode only).")
    p.add_argument("--save-dir",    type=str,   default=None,
                   help="Save result JSONs to this directory (directory mode).")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # ── Directory mode ───────────────────────────────────────────────
    if args.dir:
        run_directory_comparison(
            checkpoint_dir=args.dir,
            n_games=args.games,
            max_moves=args.max_moves,
            temperature=args.temperature,
            device=args.device,
            save_dir=args.save_dir,
        )
        return

    # ── Two-checkpoint mode ──────────────────────────────────────────
    if not args.model_a or not args.model_b:
        print("  ERROR: provide --model-a and --model-b, or use --dir.")
        parser.print_help()
        sys.exit(1)

    for path in (args.model_a, args.model_b):
        if not os.path.exists(path):
            print(f"  ERROR: checkpoint not found: {path}")
            sys.exit(1)

    model_a = PolicyNetwork()
    epoch_a = load_checkpoint(args.model_a, model_a)
    model_a.to(args.device)

    model_b = PolicyNetwork()
    epoch_b = load_checkpoint(args.model_b, model_b)
    model_b.to(args.device)

    print(f"\n{'='*60}")
    print(f"  Chess RL Agent — Checkpoint Comparison")
    print(f"{'='*60}")
    print(f"  Model A : {args.model_a}  (epoch {epoch_a})")
    print(f"  Model B : {args.model_b}  (epoch {epoch_b})")
    print(f"  Games   : {args.games}")
    print(f"  Device  : {args.device}")
    print(f"{'='*60}")

    result = run_match(
        model_a=model_a, model_b=model_b,
        model_a_path=args.model_a, model_b_path=args.model_b,
        model_a_epoch=epoch_a, model_b_epoch=epoch_b,
        n_games=args.games, max_moves=args.max_moves,
        temperature=args.temperature, device=args.device,
        verbose=args.verbose,
    )

    print(result.summary())

    if args.save_path:
        result.save_json(args.save_path)
        print(f"  Result saved: {args.save_path}")

    # Exit 0 if A wins, 1 if B wins, 2 if draw (useful for shell scripts)
    sys.exit(0 if result.a_winrate > 0.5 else (2 if result.a_winrate == 0.5 else 1))


if __name__ == "__main__":
    main()