"""
src/training/pgn_writer.py
--------------------------
Convert self-play GameRecord objects to PGN format.

PGN (Portable Game Notation) is the universal standard for recording chess
games.  Any PGN file produced here can be loaded into:

  - https://lichess.org/analysis   (paste PGN to replay move-by-move)
  - https://www.chess.com/analysis
  - Any desktop GUI: Arena, Fritz, Scid, HIARCS Chess Explorer

How move reconstruction works
------------------------------
Each GameSample stores:
    legal_moves  : list[chess.Move]  — all legal moves at that position
    move_index   : int               — which move was chosen

We replay from the starting position, and at each step pick
    legal_moves[move_index]
to recover the exact move played.  This reconstructs the full game tree
without needing to store a separate board history.

Public API
----------
    records_to_pgn(records, epoch, white_label, black_label) -> str
        Convert a list of GameRecord objects to a multi-game PGN string.

    save_pgn(pgn_text, epoch, pgn_dir) -> str
        Write the PGN string to disk and return the file path.
"""

from __future__ import annotations

import os
from datetime import date
from typing import TYPE_CHECKING

import chess
import chess.pgn

if TYPE_CHECKING:
    from src.training.self_play import GameRecord


# ---------------------------------------------------------------------------
# Result map
# ---------------------------------------------------------------------------

_RESULT_TO_PGN: dict[str, str] = {
    "white_wins"       : "1-0",
    "black_wins"       : "0-1",
    "draw"             : "1/2-1/2",
    "max_moves_reached": "1/2-1/2",   # capped games count as draws
    "in_progress"      : "*",
}

_RESULT_TO_TERMINATION: dict[str, str] = {
    "white_wins"       : "White wins by checkmate",
    "black_wins"       : "Black wins by checkmate",
    "draw"             : "Draw",
    "max_moves_reached": "Adjudicated — move limit reached",
    "in_progress"      : "Unterminated",
}


# ---------------------------------------------------------------------------
# Single-game reconstruction
# ---------------------------------------------------------------------------

def _record_to_game(
    record      : "GameRecord",
    game_number : int,
    epoch       : int,
    white_label : str,
    black_label : str,
) -> chess.pgn.Game:
    """
    Reconstruct one chess.pgn.Game from a GameRecord.

    Parameters
    ----------
    record      : GameRecord    The self-play game record.
    game_number : int           1-based index within the epoch's batch.
    epoch       : int           Training epoch (used in headers).
    white_label : str           Name tag for the White player.
    black_label : str           Name tag for the Black player.

    Returns
    -------
    chess.pgn.Game
    """
    game         = chess.pgn.Game()
    today        = date.today().isoformat()
    pgn_result   = _RESULT_TO_PGN.get(record.result, "*")
    termination  = _RESULT_TO_TERMINATION.get(record.result, "Unknown")

    # Resolve opponent label from the record's opponent_type tag
    opp_type = getattr(record, "opponent_type", "self")
    if opp_type == "self":
        event_label = f"Self-Play Epoch {epoch}"
        black_tag   = black_label
    else:
        event_label = f"vs {opp_type} Epoch {epoch}"
        black_tag   = opp_type   # e.g. "RandomAgent" or "CheckpointAgent"

    # PGN headers
    game.headers["Event"]        = event_label
    game.headers["Site"]         = "Chess RL Agent"
    game.headers["Date"]         = today
    game.headers["Round"]        = str(game_number)
    game.headers["White"]        = white_label
    game.headers["Black"]        = black_tag
    game.headers["Result"]       = pgn_result
    game.headers["Termination"]  = termination
    game.headers["OpponentType"] = opp_type
    game.headers["WhiteElo"]     = "?"
    game.headers["BlackElo"]     = "?"

    # Replay moves from samples
    board = chess.Board()
    node  = game

    for sample in record.samples:
        # Guard: legal_moves may be empty if the game record is corrupt
        if not sample.legal_moves:
            break

        # Recover the chosen move
        idx  = sample.move_index
        if idx < 0 or idx >= len(sample.legal_moves):
            break   # index out of range — stop here, don't crash

        move = sample.legal_moves[idx]

        # Validate move is still legal in the replayed position
        if move not in board.legal_moves:
            break

        node = node.add_variation(move)
        board.push(move)

    game.headers["Result"] = pgn_result
    return game


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------

def records_to_pgn(
    records     : list["GameRecord"],
    epoch       : int,
    white_label : str = "ChessRL",
    black_label : str = "ChessRL",
) -> str:
    """
    Convert a list of GameRecord objects to a multi-game PGN string.

    Parameters
    ----------
    records     : list[GameRecord]   Self-play games from one epoch.
    epoch       : int                Training epoch number.
    white_label : str                PGN White player name.
    black_label : str                PGN Black player name.

    Returns
    -------
    str   Full PGN text (multiple games separated by blank lines).
    """
    games_pgn: list[str] = []

    for i, record in enumerate(records, start=1):
        try:
            game = _record_to_game(
                record=record,
                game_number=i,
                epoch=epoch,
                white_label=white_label,
                black_label=black_label,
            )
            exporter = chess.pgn.StringExporter(
                headers=True,
                variations=False,
                comments=False,
            )
            games_pgn.append(game.accept(exporter))
        except Exception:
            # Never crash training due to a PGN serialisation error
            continue

    return "\n\n".join(games_pgn)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_pgn(
    pgn_text : str,
    epoch    : int,
    pgn_dir  : str,
) -> str:
    """
    Write a PGN string to disk.

    File name:  logs/games/games_epoch_NNNN.pgn

    Parameters
    ----------
    pgn_text : str   PGN content to write.
    epoch    : int   Training epoch (zero-padded in filename).
    pgn_dir  : str   Directory to write into.

    Returns
    -------
    str   Absolute path of the written file.
    """
    os.makedirs(pgn_dir, exist_ok=True)
    filename = f"games_epoch_{epoch:04d}.pgn"
    path     = os.path.join(pgn_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(pgn_text)
        f.write("\n")   # trailing newline — some PGN readers require it

    return path