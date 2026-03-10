"""
scripts/save_game_pgn.py
------------------------
Standalone PGN game saver — replay any checkpoint and export games.

Loads a trained model checkpoint, runs N self-play games, and writes the
results as a PGN file that can be replayed on any chess platform.

Paste the PGN into:
  https://lichess.org/analysis      ← recommended, free, no login needed
  https://www.chess.com/analysis
  Any desktop GUI (Arena, Fritz, Scid)

Use this to visually inspect how the model plays at different training
stages — look for improvements in piece development, avoiding blunders,
and steering toward checkmate.

Usage
-----
    # Save 20 games from a checkpoint
    python scripts/save_game_pgn.py \\
        --model models/policy_epoch_0080.pt \\
        --games 20

    # Specify output path
    python scripts/save_game_pgn.py \\
        --model models/policy_epoch_0080.pt \\
        --output logs/games/epoch80_showcase.pgn

    # Compare play at two epochs (saves separate files)
    python scripts/save_game_pgn.py \\
        --model models/policy_epoch_0020.pt \\
        --output logs/games/early.pgn
    python scripts/save_game_pgn.py \\
        --model models/policy_epoch_0080.pt \\
        --output logs/games/late.pgn

    # Generate a PGN from every checkpoint in a directory
    python scripts/save_game_pgn.py --dir models/ --games 10

    # Use the random agent as opponent (model plays White)
    python scripts/save_game_pgn.py \\
        --model models/policy_epoch_0080.pt \\
        --vs-random --games 20
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chess
import chess.pgn
import torch
from datetime import date

from src.environment.board_encoder import encode_board
from src.evaluation.random_agent import RandomAgent
from src.models.policy_network import PolicyNetwork
from src.training.train_policy import load_checkpoint
from src.training.pgn_writer import records_to_pgn, save_pgn


# ---------------------------------------------------------------------------
# Self-play game runner (lightweight — no training samples needed)
# ---------------------------------------------------------------------------

def _select_move(
    model       : PolicyNetwork,
    board       : chess.Board,
    temperature : float,
    device      : str,
) -> chess.Move:
    """Run the policy network and return one selected move."""
    legal   = list(board.legal_moves)
    state_t = torch.tensor(
        encode_board(board), dtype=torch.float32, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        return model.select_move(state_t, legal, temperature=temperature)


def _play_self_play_game(
    model     : PolicyNetwork,
    max_moves : int,
    temp      : float,
    device    : str,
) -> tuple[chess.Board, str]:
    """
    Play one model-vs-self game.

    Returns
    -------
    board  : chess.Board   Final board state (with full move stack).
    result : str           'white_wins', 'black_wins', 'draw', or 'max_moves_reached'.
    """
    board = chess.Board()

    for _ in range(max_moves):
        if board.is_game_over() or not list(board.legal_moves):
            break
        move = _select_move(model, board, temp, device)
        board.push(move)

    if not board.is_game_over():
        return board, "max_moves_reached"

    res = board.result()
    if res == "1-0":
        return board, "white_wins"
    elif res == "0-1":
        return board, "black_wins"
    else:
        return board, "draw"


def _play_vs_random_game(
    model     : PolicyNetwork,
    agent     : RandomAgent,
    max_moves : int,
    temp      : float,
    device    : str,
    model_white: bool,
) -> tuple[chess.Board, str]:
    """
    Play one game between the model and the random agent.

    Returns the final board and result from the model's perspective.
    """
    board        = chess.Board()
    model_colour = chess.WHITE if model_white else chess.BLACK

    for _ in range(max_moves):
        if board.is_game_over() or not list(board.legal_moves):
            break
        if board.turn == model_colour:
            move = _select_move(model, board, temp, device)
        else:
            move = agent.select_move(board)
        board.push(move)

    if not board.is_game_over():
        return board, "max_moves_reached"

    res = board.result()
    if res == "1-0":
        return board, "white_wins"
    elif res == "0-1":
        return board, "black_wins"
    else:
        return board, "draw"


# ---------------------------------------------------------------------------
# Board → PGN conversion (direct from board.move_stack, not GameRecord)
# ---------------------------------------------------------------------------

_RESULT_TO_PGN = {
    "white_wins"       : "1-0",
    "black_wins"       : "0-1",
    "draw"             : "1/2-1/2",
    "max_moves_reached": "1/2-1/2",
}

_RESULT_TO_TERMINATION = {
    "white_wins"       : "White wins by checkmate",
    "black_wins"       : "Black wins by checkmate",
    "draw"             : "Draw",
    "max_moves_reached": "Adjudicated — move limit reached",
}


def board_to_pgn_game(
    board       : chess.Board,
    result      : str,
    game_number : int,
    white_label : str,
    black_label : str,
    event       : str,
) -> chess.pgn.Game:
    """Convert a finished chess.Board (with move_stack) to a chess.pgn.Game."""
    game = chess.pgn.Game()
    game.headers["Event"]       = event
    game.headers["Site"]        = "Chess RL Agent"
    game.headers["Date"]        = date.today().isoformat()
    game.headers["Round"]       = str(game_number)
    game.headers["White"]       = white_label
    game.headers["Black"]       = black_label
    game.headers["Result"]      = _RESULT_TO_PGN.get(result, "*")
    game.headers["Termination"] = _RESULT_TO_TERMINATION.get(result, "Unknown")

    # chess.pgn.Game.from_board() is the simplest way to convert
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"]  = event
    game.headers["Site"]   = "Chess RL Agent"
    game.headers["Date"]   = date.today().isoformat()
    game.headers["Round"]  = str(game_number)
    game.headers["White"]  = white_label
    game.headers["Black"]  = black_label

    return game


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def generate_pgn(
    model      : PolicyNetwork,
    epoch      : int,
    n_games    : int,
    max_moves  : int,
    temperature: float,
    device     : str,
    vs_random  : bool,
) -> str:
    """
    Run n_games and return a multi-game PGN string.

    Parameters
    ----------
    vs_random : bool
        If True, model plays against the random agent (alternating sides).
        If False, model plays against itself.
    """
    agent      = RandomAgent() if vs_random else None
    games_pgn  : list[str] = []

    wins = losses = draws = capped = 0
    event = f"Self-Play Epoch {epoch}" if not vs_random else f"Model vs Random Epoch {epoch}"

    for i in range(n_games):
        if vs_random:
            model_white = (i % 2 == 0)
            white_label = "ChessRL" if model_white else "Random"
            black_label = "Random"  if model_white else "ChessRL"
            board, result = _play_vs_random_game(
                model, agent, max_moves, temperature, device, model_white
            )
        else:
            white_label = "ChessRL-White"
            black_label = "ChessRL-Black"
            board, result = _play_self_play_game(model, max_moves, temperature, device)

        # Tally
        if   result == "white_wins":        wins   += 1
        elif result == "black_wins":        losses += 1
        elif result == "max_moves_reached": capped += 1
        else:                               draws  += 1

        game = board_to_pgn_game(
            board=board, result=result,
            game_number=i + 1,
            white_label=white_label, black_label=black_label,
            event=event,
        )
        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        games_pgn.append(game.accept(exporter))

        # Progress
        print(
            f"\r  Game {i+1:>4}/{n_games}  "
            f"W {wins}  L {losses}  D {draws}  Cap {capped}",
            end="", flush=True
        )

    print()  # newline after progress
    return "\n\n".join(games_pgn)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate PGN files from a Chess RL model checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Source
    p.add_argument("--model", type=str, default=None,
                   help="Path to a single checkpoint .pt file.")
    p.add_argument("--dir",   type=str, default=None,
                   help="Process every policy_epoch_*.pt checkpoint in this directory.")

    # Game settings
    p.add_argument("--games",       type=int,   default=20,
                   help="Number of games to generate per checkpoint.")
    p.add_argument("--max-moves",   type=int,   default=200,
                   help="Move cap per game.")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="Sampling temperature (lower = more deterministic).")
    p.add_argument("--vs-random",   action="store_true",
                   help="Play model vs random agent instead of self-play.")

    # Output
    p.add_argument("--output",  type=str, default=None,
                   help="Output PGN path (single-model mode). "
                        "Defaults to logs/games/games_epoch_NNNN.pgn.")
    p.add_argument("--out-dir", type=str, default="logs/games",
                   help="Output directory (directory mode).")

    # Hardware
    p.add_argument("--device", type=str, default="cpu",
                   help="Torch device ('cpu' or 'cuda').")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if not args.model and not args.dir:
        print("  ERROR: provide --model or --dir.")
        parser.print_help()
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Chess RL Agent — PGN Game Saver")
    print(f"{'='*60}")

    # ── Resolve checkpoints ──────────────────────────────────────────
    if args.dir:
        pattern = os.path.join(args.dir, "policy_epoch_*.pt")
        paths   = sorted(glob.glob(pattern))
        if not paths:
            print(f"  ERROR: No policy_epoch_*.pt files in '{args.dir}'.")
            sys.exit(1)
    else:
        if not os.path.exists(args.model):
            print(f"  ERROR: checkpoint not found: {args.model}")
            sys.exit(1)
        paths = [args.model]

    for ckpt_path in paths:
        model = PolicyNetwork()
        epoch = load_checkpoint(ckpt_path, model)
        model.to(args.device)
        model.eval()

        print(f"\n  Checkpoint : {ckpt_path}  (epoch {epoch})")
        print(f"  Games      : {args.games}")
        print(f"  Opponent   : {'Random Agent' if args.vs_random else 'Self-play'}")
        print(f"  Max moves  : {args.max_moves}")
        print()

        pgn_text = generate_pgn(
            model=model, epoch=epoch,
            n_games=args.games, max_moves=args.max_moves,
            temperature=args.temperature, device=args.device,
            vs_random=args.vs_random,
        )

        # Determine output path
        if args.model and args.output:
            out_path = args.output
        else:
            os.makedirs(args.out_dir, exist_ok=True)
            out_path = os.path.join(args.out_dir, f"games_epoch_{epoch:04d}.pgn")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(pgn_text)
            f.write("\n")

        print(f"\n  PGN saved  : {out_path}")
        print(f"  Games      : {pgn_text.count('[Event ')}")
        print()
        print("  To replay: paste into https://lichess.org/analysis")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()