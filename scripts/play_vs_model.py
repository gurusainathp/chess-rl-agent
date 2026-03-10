"""
scripts/play_vs_model.py
------------------------
Interactive terminal interface — play chess against the trained RL agent.

The human enters moves in SAN notation (e.g. e4, Nf3, O-O) or UCI notation
(e.g. e2e4, g1f3).  SAN is tried first; if parsing fails, UCI is attempted.

The board is printed to the terminal after every move using Unicode chess
pieces for readability.

Usage
-----
    # Play against the latest checkpoint
    python scripts/play_vs_model.py --model data/models/policy_epoch_0050.pt

    # Choose your side interactively (default) or set it upfront
    python scripts/play_vs_model.py --model data/models/policy_epoch_0050.pt --side white
    python scripts/play_vs_model.py --model data/models/policy_epoch_0050.pt --side black

    # Adjust model temperature (lower = stronger/more deterministic)
    python scripts/play_vs_model.py --model data/models/policy_epoch_0050.pt --temperature 0.05

    # Play without a trained model (random agent vs random agent — for testing)
    python scripts/play_vs_model.py --random

Commands during play
--------------------
    resign   — forfeit the current game
    draw     — claim a draw (always accepted)
    moves    — print all legal moves in the current position
    undo     — take back the last two half-moves (your move + model's reply)
    quit     — exit the program
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chess
import torch

from src.environment.board_encoder import encode_board
from src.evaluation.random_agent import RandomAgent
from src.models.policy_network import PolicyNetwork
from src.training.train_policy import load_checkpoint


# ---------------------------------------------------------------------------
# Unicode board renderer
# ---------------------------------------------------------------------------

# Piece letters: uppercase = White, lowercase = Black (standard chess notation)
_PIECE_LETTERS = {
    (chess.PAWN,   chess.WHITE): "P",
    (chess.KNIGHT, chess.WHITE): "N",
    (chess.BISHOP, chess.WHITE): "B",
    (chess.ROOK,   chess.WHITE): "R",
    (chess.QUEEN,  chess.WHITE): "Q",
    (chess.KING,   chess.WHITE): "K",
    (chess.PAWN,   chess.BLACK): "p",
    (chess.KNIGHT, chess.BLACK): "n",
    (chess.BISHOP, chess.BLACK): "b",
    (chess.ROOK,   chess.BLACK): "r",
    (chess.QUEEN,  chess.BLACK): "q",
    (chess.KING,   chess.BLACK): "k",
}


def render_board(board: chess.Board, human_is_white: bool) -> str:
    """
    Render the board as a clean fixed-width ASCII grid.

    Works correctly on all terminals including Windows PowerShell.

    Layout (human plays White):
        8 | r n b q k b n r |
        7 | p p p p p p p p |
        ...
        1 | R N B Q K B N R |
            a b c d e f g h

    Uppercase letters = White pieces, lowercase = Black pieces.
    Dots (.) mark empty squares.
    Light/dark squares shown by spacing only — no ANSI colours
    so alignment is always perfect regardless of terminal support.
    """
    ranks = range(7, -1, -1) if human_is_white else range(0, 8)
    files = range(0, 8)      if human_is_white else range(7, -1, -1)
    file_chars = "abcdefgh" if human_is_white else "hgfedcba"

    border = "  +" + "---+" * 8
    lines  = ["", border]

    for rank in ranks:
        cells = []
        for file in files:
            square = chess.square(file, rank)
            piece  = board.piece_at(square)
            letter = _PIECE_LETTERS.get((piece.piece_type, piece.color), ".") if piece else "."
            cells.append(f" {letter} ")
        row = f"{rank + 1} |" + "|".join(cells) + "|"
        lines.append(row)
        lines.append(border)

    file_label = "    " + "   ".join(file_chars)
    lines.append(file_label)
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Move parsing
# ---------------------------------------------------------------------------

def parse_move(text: str, board: chess.Board) -> chess.Move | None:
    """
    Parse a move string — try SAN first, then UCI.

    Returns None if parsing fails or the move is illegal.
    """
    text = text.strip()
    if not text:
        return None

    # Try SAN (e.g. e4, Nf3, O-O, Qxd5+)
    try:
        move = board.parse_san(text)
        if move in board.legal_moves:
            return move
    except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
        pass

    # Try UCI (e.g. e2e4, g1f3, e7e8q)
    try:
        move = chess.Move.from_uci(text)
        if move in board.legal_moves:
            return move
    except (chess.InvalidMoveError, ValueError):
        pass

    return None


# ---------------------------------------------------------------------------
# Model move selection
# ---------------------------------------------------------------------------

def model_move(
    model      : PolicyNetwork | None,
    board      : chess.Board,
    temperature: float,
    device     : str,
    fallback   : RandomAgent,
) -> chess.Move:
    """
    Select the model's move.  Falls back to the random agent if model is None.
    """
    legal = list(board.legal_moves)

    if model is None:
        return fallback.select_move(board)

    state_t = torch.tensor(
        encode_board(board), dtype=torch.float32, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        return model.select_move(state_t, legal, temperature=temperature)


# ---------------------------------------------------------------------------
# Single game loop
# ---------------------------------------------------------------------------

def play_game(
    model        : PolicyNetwork | None,
    human_is_white: bool,
    temperature  : float,
    device       : str,
) -> None:
    """Run one interactive game between the human and the model."""
    board    = chess.Board()
    fallback = RandomAgent()
    history  : list[chess.Move] = []   # for undo support

    human_colour = chess.WHITE if human_is_white else chess.BLACK
    model_colour_str = "Black" if human_is_white else "White"
    human_colour_str = "White" if human_is_white else "Black"

    print(f"\n  You are playing as {human_colour_str}.")
    print(f"  Model is playing as {model_colour_str}.")
    print(f"  Enter moves in SAN (e4, Nf3) or UCI (e2e4, g1f3) notation.")
    print(f"  Type 'moves' to see legal moves, 'resign', 'draw', 'undo', or 'quit'.\n")

    while not board.is_game_over():

        print(render_board(board, human_is_white))

        # Status line
        turn_str = "White" if board.turn == chess.WHITE else "Black"
        print(f"  {'─'*40}")
        if board.is_check():
            print(f"  {turn_str} is in CHECK")
        print(f"  Move {board.fullmove_number} — {turn_str} to move")

        # ── Human's turn ──────────────────────────────────────────────
        if board.turn == human_colour:
            while True:
                try:
                    raw = input("  Your move: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n  Interrupted — exiting.")
                    return

                # Commands are checked case-insensitively.
                # The move itself is kept in its original case because SAN
                # is case-sensitive: 'Nc3' != 'nc3'.
                cmd = raw.lower()

                if cmd == "quit":
                    print("  Goodbye!")
                    sys.exit(0)

                if cmd == "resign":
                    print(f"\n  You resigned.  The model wins.")
                    return

                if cmd == "draw":
                    print(f"\n  Draw claimed.  Game over.")
                    return

                if cmd == "moves":
                    legal_sans = sorted(board.san(m) for m in board.legal_moves)
                    print(f"  Legal moves ({len(legal_sans)}): {', '.join(legal_sans)}")
                    continue

                if cmd == "undo":
                    if len(history) < 2:
                        print("  Nothing to undo yet.")
                        continue
                    board.pop(); history.pop()
                    board.pop(); history.pop()
                    print("  Last two half-moves undone.")
                    break   # re-print the board

                move = parse_move(raw, board)   # raw preserves original case
                if move is None:
                    print(f"  '{raw}' is not a valid or legal move — try again.")
                    continue

                board.push(move)
                history.append(move)
                break

        # ── Model's turn ──────────────────────────────────────────────
        else:
            print(f"  Model is thinking…")
            move = model_move(model, board, temperature, device, fallback)
            san  = board.san(move)
            board.push(move)
            history.append(move)
            print(f"  Model plays: {san}  ({move.uci()})")

    # ── Game over ─────────────────────────────────────────────────────
    print(render_board(board, human_is_white))
    print(f"\n  {'='*40}")
    result = board.result()   # "1-0", "0-1", "1/2-1/2"

    if result == "1/2-1/2":
        print("  Result: Draw")
    elif (result == "1-0" and human_is_white) or (result == "0-1" and not human_is_white):
        print("  Result: You win! 🎉")
    else:
        print("  Result: Model wins.")

    print(f"  {'='*40}\n")
    print(f"  PGN movetext: {board.variation_san(board.move_stack[0:])}" 
          if board.move_stack else "")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Play chess against the trained RL agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", type=str, default=None,
        help="Path to a model checkpoint .pt file.  Omit to play vs random agent.",
    )
    p.add_argument(
        "--side", type=str, default=None, choices=["white", "black"],
        help="Your colour.  If omitted, you are asked at the start.",
    )
    p.add_argument(
        "--temperature", type=float, default=0.1,
        help="Model sampling temperature (lower = stronger/more deterministic).",
    )
    p.add_argument(
        "--random", action="store_true",
        help="Play against the random agent regardless of --model.",
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

    # ── Banner ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Chess RL Agent — Play vs Model")
    print(f"{'='*60}")

    # ── Load model ───────────────────────────────────────────────────
    model = None

    if not args.random and args.model is not None:
        if not os.path.exists(args.model):
            print(f"  ERROR: checkpoint not found: {args.model}")
            sys.exit(1)
        model = PolicyNetwork()
        epoch = load_checkpoint(args.model, model)
        model.to(args.device)
        model.eval()
        print(f"  Model loaded    : {args.model}  (epoch {epoch})")
        print(f"  Parameters      : {model.count_parameters():,}")
        print(f"  Temperature     : {args.temperature}")
    else:
        print("  Opponent        : Random Agent")

    print(f"{'='*60}")

    # ── Game loop ────────────────────────────────────────────────────
    while True:
        # Determine human side
        if args.side is not None:
            human_is_white = (args.side == "white")
        else:
            while True:
                try:
                    choice = input("\n  Play as [W]hite or [B]lack? ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\n  Exiting.")
                    sys.exit(0)
                if choice in ("w", "white"):
                    human_is_white = True
                    break
                if choice in ("b", "black"):
                    human_is_white = False
                    break
                print("  Please enter 'w' or 'b'.")

        play_game(
            model=model,
            human_is_white=human_is_white,
            temperature=args.temperature,
            device=args.device,
        )

        # Play again?
        try:
            again = input("  Play again? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            sys.exit(0)

        if again not in ("y", "yes"):
            print("\n  Thanks for playing! Goodbye.\n")
            break


if __name__ == "__main__":
    main()