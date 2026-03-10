"""
scripts/elo_rating.py
---------------------
Compute and track Elo ratings for all model checkpoints.

Assigns every checkpoint an Elo rating by playing it against a set of
reference opponents.  The random agent is anchored at Elo 800 — a
commonly used baseline for very weak play.

How Elo works here
------------------
  1. All checkpoints start at Elo 800 (same as random).
  2. Each checkpoint plays a match against the random agent (always
     available) and optionally against adjacent checkpoints.
  3. Ratings are updated after each match using the standard Elo formula.
  4. Results are printed as a rating ladder and optionally saved to JSON
     and a PNG plot.

Standard Elo formula
--------------------
  Expected score  E_A = 1 / (1 + 10^((R_B - R_A) / 400))
  New rating      R_A' = R_A + K * (S_A - E_A)

  where S_A is the actual score (1=win, 0.5=draw, 0=loss)
  and   K   is the K-factor (sensitivity of rating changes).

Usage
-----
    # Rate all checkpoints in models/ against the random agent
    python scripts/elo_rating.py --dir models/

    # Rate a specific checkpoint
    python scripts/elo_rating.py --model models/policy_epoch_0080.pt

    # Also play checkpoints against each other (more accurate)
    python scripts/elo_rating.py --dir models/ --vs-each-other

    # Save the rating table to JSON and PNG
    python scripts/elo_rating.py --dir models/ \\
        --save-json results/elo.json \\
        --save-plot results/elo_curve.png

    # Adjust games per match and K-factor
    python scripts/elo_rating.py --dir models/ --games 100 --k-factor 32
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chess
import torch

from src.environment.board_encoder import encode_board
from src.evaluation.random_agent import RandomAgent
from src.models.policy_network import PolicyNetwork
from src.training.train_policy import load_checkpoint


# ---------------------------------------------------------------------------
# Elo maths
# ---------------------------------------------------------------------------

RANDOM_AGENT_ELO  = 800    # anchor — random play is roughly 800 Elo
DEFAULT_START_ELO = 800    # new models start here
DEFAULT_K         = 32     # standard K-factor for non-established players


def expected_score(rating_a: float, rating_b: float) -> float:
    """Return Model A's expected score against Model B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def updated_elo(
    rating    : float,
    opponent  : float,
    actual    : float,   # 1.0=win, 0.5=draw, 0.0=loss
    k_factor  : float = DEFAULT_K,
) -> float:
    """Return the updated Elo after one match result."""
    e = expected_score(rating, opponent)
    return rating + k_factor * (actual - e)


def winrate_to_elo_diff(winrate: float) -> float:
    """
    Convert a win fraction against the random agent to an Elo difference.

    Based on the logistic Elo formula.  winrate=0.5 → diff=0.
    Clamped to avoid log(0).
    """
    winrate = max(0.001, min(0.999, winrate))
    return -400.0 * math.log10(1.0 / winrate - 1.0)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PlayerRecord:
    """Rating record for one checkpoint or the random agent."""
    name      : str
    path      : str | None   # None for the random agent
    epoch     : int | None   # None for the random agent
    elo       : float        = DEFAULT_START_ELO
    wins      : int          = 0
    losses    : int          = 0
    draws     : int          = 0
    matches   : int          = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def winrate(self) -> float:
        return self.wins / self.games if self.games > 0 else 0.0

    def record_result(self, score: float, new_elo: float) -> None:
        self.elo = new_elo
        self.matches += 1
        if score == 1.0:
            self.wins += 1
        elif score == 0.0:
            self.losses += 1
        else:
            self.draws += 1


# ---------------------------------------------------------------------------
# Move selection helpers
# ---------------------------------------------------------------------------

def _model_move(
    model      : PolicyNetwork,
    board      : chess.Board,
    temperature: float,
    device     : str,
) -> chess.Move:
    legal   = list(board.legal_moves)
    state_t = torch.tensor(
        encode_board(board), dtype=torch.float32, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        return model.select_move(state_t, legal, temperature=temperature)


def _play_game_model_vs_random(
    model       : PolicyNetwork,
    agent       : RandomAgent,
    model_white : bool,
    max_moves   : int,
    temperature : float,
    device      : str,
) -> float:
    """
    Play one game and return the model's score (1=win, 0.5=draw, 0=loss).
    """
    board = chess.Board()
    model_colour = chess.WHITE if model_white else chess.BLACK

    for _ in range(max_moves):
        if board.is_game_over() or not list(board.legal_moves):
            break
        if board.turn == model_colour:
            move = _model_move(model, board, temperature, device)
        else:
            move = agent.select_move(board)
        board.push(move)

    if not board.is_game_over():
        return 0.5   # capped → draw

    res = board.result()
    if res == "1/2-1/2":
        return 0.5
    if (res == "1-0" and model_white) or (res == "0-1" and not model_white):
        return 1.0
    return 0.0


def _play_game_model_vs_model(
    model_a     : PolicyNetwork,
    model_b     : PolicyNetwork,
    a_is_white  : bool,
    max_moves   : int,
    temperature : float,
    device      : str,
) -> float:
    """
    Play one game between two models.
    Returns Model A's score (1=win, 0.5=draw, 0=loss).
    """
    board = chess.Board()
    a_colour = chess.WHITE if a_is_white else chess.BLACK

    for _ in range(max_moves):
        if board.is_game_over() or not list(board.legal_moves):
            break
        if board.turn == a_colour:
            move = _model_move(model_a, board, temperature, device)
        else:
            move = _model_move(model_b, board, temperature, device)
        board.push(move)

    if not board.is_game_over():
        return 0.5

    res = board.result()
    if res == "1/2-1/2":
        return 0.5
    if (res == "1-0" and a_is_white) or (res == "0-1" and not a_is_white):
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Match runners
# ---------------------------------------------------------------------------

def match_vs_random(
    record      : PlayerRecord,
    model       : PolicyNetwork,
    agent       : RandomAgent,
    n_games     : int,
    max_moves   : int,
    temperature : float,
    k_factor    : float,
    device      : str,
    random_elo  : float,
) -> None:
    """Play n_games vs random, update record.elo in place."""
    total_score = 0.0
    for i in range(n_games):
        model_white = (i % 2 == 0)
        score = _play_game_model_vs_random(
            model, agent, model_white, max_moves, temperature, device
        )
        total_score += score

    avg_score   = total_score / n_games
    new_elo     = updated_elo(record.elo, random_elo, avg_score, k_factor)
    record.record_result(avg_score, new_elo)


def match_vs_model(
    record_a    : PlayerRecord,
    model_a     : PolicyNetwork,
    record_b    : PlayerRecord,
    model_b     : PolicyNetwork,
    n_games     : int,
    max_moves   : int,
    temperature : float,
    k_factor    : float,
    device      : str,
) -> None:
    """Play n_games between two models, update both records in place."""
    total_a = 0.0
    for i in range(n_games):
        a_is_white = (i % 2 == 0)
        score_a = _play_game_model_vs_model(
            model_a, model_b, a_is_white, max_moves, temperature, device
        )
        total_a += score_a

    avg_a   = total_a / n_games
    avg_b   = 1.0 - avg_a

    new_a = updated_elo(record_a.elo, record_b.elo, avg_a, k_factor)
    new_b = updated_elo(record_b.elo, record_a.elo, avg_b, k_factor)

    record_a.record_result(avg_a, new_a)
    record_b.record_result(avg_b, new_b)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_leaderboard(records: list[PlayerRecord], random_record: PlayerRecord) -> None:
    """Print the Elo leaderboard sorted by rating descending."""
    all_records = sorted(
        [random_record] + records, key=lambda r: r.elo, reverse=True
    )
    print(f"\n  {'Rank':<5} {'Name':<30} {'Elo':>6}  {'W':>4} {'L':>4} {'D':>4}  {'Winrate':>8}")
    print(f"  {'─'*70}")
    for rank, r in enumerate(all_records, 1):
        name = r.name[:28]
        wr   = f"{r.winrate*100:.1f}%" if r.games > 0 else "—"
        print(
            f"  {rank:<5} {name:<30} {r.elo:>6.0f}  "
            f"{r.wins:>4} {r.losses:>4} {r.draws:>4}  {wr:>8}"
        )
    print()


def save_leaderboard_json(
    records       : list[PlayerRecord],
    random_record : PlayerRecord,
    path          : str,
) -> None:
    data = {
        "random_agent": {
            "name": random_record.name,
            "elo":  random_record.elo,
        },
        "checkpoints": [
            {
                "name"    : r.name,
                "path"    : r.path,
                "epoch"   : r.epoch,
                "elo"     : round(r.elo, 1),
                "wins"    : r.wins,
                "losses"  : r.losses,
                "draws"   : r.draws,
                "winrate" : round(r.winrate, 4),
            }
            for r in sorted(records, key=lambda r: r.elo, reverse=True)
        ],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Leaderboard saved: {path}")


def save_elo_plot(
    records       : list[PlayerRecord],
    random_record : PlayerRecord,
    path          : str,
) -> None:
    """Plot Elo vs epoch and save to PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping plot.  pip install matplotlib")
        return

    # Sort by epoch
    sorted_records = sorted(
        [r for r in records if r.epoch is not None],
        key=lambda r: r.epoch
    )
    if not sorted_records:
        print("  No checkpoint records to plot.")
        return

    epochs = [r.epoch for r in sorted_records]
    elos   = [r.elo   for r in sorted_records]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, elos, marker="o", linewidth=2, markersize=5, label="Model Elo")
    ax.axhline(
        random_record.elo, color="red", linestyle="--",
        linewidth=1.5, label=f"Random Agent ({random_record.elo:.0f})"
    )
    ax.fill_between(epochs, random_record.elo, elos,
                    where=[e > random_record.elo for e in elos],
                    alpha=0.15, color="green", label="Above random")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Elo Rating")
    ax.set_title("Chess RL Agent — Elo Rating Progression")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Elo plot saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute Elo ratings for Chess RL model checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",  type=str, default=None,
                   help="Single checkpoint to rate.")
    p.add_argument("--dir",    type=str, default=None,
                   help="Directory containing policy_epoch_*.pt checkpoints.")
    p.add_argument("--games",         type=int,   default=50,
                   help="Games per match (vs random agent).")
    p.add_argument("--max-moves",     type=int,   default=200,
                   help="Move cap per game.")
    p.add_argument("--temperature",   type=float, default=0.1,
                   help="Sampling temperature for all models.")
    p.add_argument("--k-factor",      type=float, default=DEFAULT_K,
                   help="Elo K-factor (sensitivity of rating changes).")
    p.add_argument("--random-elo",    type=float, default=RANDOM_AGENT_ELO,
                   help="Fixed Elo assigned to the random agent.")
    p.add_argument("--vs-each-other", action="store_true",
                   help="Also play adjacent checkpoints against each other.")
    p.add_argument("--device",        type=str,   default="cpu",
                   help="Torch device.")
    p.add_argument("--save-json",     type=str,   default=None,
                   help="Save leaderboard to JSON.")
    p.add_argument("--save-plot",     type=str,   default=None,
                   help="Save Elo-vs-epoch PNG to this path.")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # ── Resolve checkpoints ──────────────────────────────────────────
    checkpoint_paths: list[str] = []
    if args.dir:
        pattern = os.path.join(args.dir, "policy_epoch_*.pt")
        checkpoint_paths = sorted(glob.glob(pattern))
        if not checkpoint_paths:
            print(f"  ERROR: no policy_epoch_*.pt files found in '{args.dir}'.")
            sys.exit(1)
    elif args.model:
        if not os.path.exists(args.model):
            print(f"  ERROR: checkpoint not found: {args.model}")
            sys.exit(1)
        checkpoint_paths = [args.model]
    else:
        print("  ERROR: provide --model or --dir.")
        parser.print_help()
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Chess RL Agent — Elo Rating System")
    print(f"{'='*60}")
    print(f"  Checkpoints   : {len(checkpoint_paths)}")
    print(f"  Games/match   : {args.games}")
    print(f"  Random Elo    : {args.random_elo}")
    print(f"  K-factor      : {args.k_factor}")
    print(f"{'='*60}\n")

    # ── Build player records ─────────────────────────────────────────
    random_record = PlayerRecord(
        name="Random Agent", path=None, epoch=None, elo=args.random_elo
    )
    agent = RandomAgent()

    records: list[PlayerRecord] = []
    models:  list[PolicyNetwork] = []

    for path in checkpoint_paths:
        model = PolicyNetwork()
        epoch = load_checkpoint(path, model)
        model.to(args.device)
        model.eval()
        name = os.path.basename(path)
        records.append(PlayerRecord(name=name, path=path, epoch=epoch))
        models.append(model)
        print(f"  Loaded: {name}  (epoch {epoch})")

    print()

    # ── Rate each checkpoint vs random agent ─────────────────────────
    total_start = time.time()
    for record, model in zip(records, models):
        print(f"  Rating {record.name} vs Random ...", end=" ", flush=True)
        match_vs_random(
            record=record, model=model, agent=agent,
            n_games=args.games, max_moves=args.max_moves,
            temperature=args.temperature, k_factor=args.k_factor,
            device=args.device, random_elo=args.random_elo,
        )
        print(f"Elo → {record.elo:.0f}")

    # ── Optional: adjacent checkpoint matches ────────────────────────
    if args.vs_each_other and len(records) > 1:
        print("\n  Playing adjacent checkpoint matches ...")
        for i in range(len(records) - 1):
            ra, rb = records[i], records[i + 1]
            ma, mb = models[i], models[i + 1]
            print(f"  {ra.name} vs {rb.name} ...", end=" ", flush=True)
            match_vs_model(
                record_a=ra, model_a=ma,
                record_b=rb, model_b=mb,
                n_games=args.games, max_moves=args.max_moves,
                temperature=args.temperature, k_factor=args.k_factor,
                device=args.device,
            )
            print(f"A={ra.elo:.0f}  B={rb.elo:.0f}")

    elapsed = time.time() - total_start

    # ── Leaderboard ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Elo Leaderboard  (total time: {elapsed:.1f}s)")
    print(f"{'='*60}")
    print_leaderboard(records, random_record)

    # ── Save outputs ─────────────────────────────────────────────────
    if args.save_json:
        save_leaderboard_json(records, random_record, args.save_json)
    if args.save_plot:
        save_elo_plot(records, random_record, args.save_plot)


if __name__ == "__main__":
    main()