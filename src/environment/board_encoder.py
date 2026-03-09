"""
board_encoder.py
----------------
Converts a python-chess Board into a fixed-size numerical tensor that
the neural network can consume.

Usage
-----
    import chess
    from chess_rl_agent.environment.board_encoder import encode_board

    board  = chess.Board()
    tensor = encode_board(board)   # shape: (13, 8, 8), dtype: float32

Encoding layout
---------------
The output is a (13, 8, 8) NumPy array — 13 planes stacked depth-wise.
Planes 0–11 are binary 8×8 grids for piece positions; plane 12 encodes
whose turn it is to move.

    Plane  0  — White pawns
    Plane  1  — White knights
    Plane  2  — White bishops
    Plane  3  — White rooks
    Plane  4  — White queens
    Plane  5  — White king

    Plane  6  — Black pawns
    Plane  7  — Black knights
    Plane  8  — Black bishops
    Plane  9  — Black rooks
    Plane 10  — Black queens
    Plane 11  — Black king

    Plane 12  — Turn indicator
                All 64 cells = 1.0 if it is White's turn to move.
                All 64 cells = 0.0 if it is Black's turn to move.

A piece-plane cell is 1.0 if that piece occupies that square, 0.0 otherwise.

Square-to-index mapping
-----------------------
python-chess numbers squares 0–63 from a1 (bottom-left) to h8 (top-right):

    square index = rank * 8 + file
    rank  = square // 8     (0 = rank 1, 7 = rank 8)
    file  = square % 8      (0 = a-file, 7 = h-file)

We map this directly onto the (row, col) axes of the 8×8 grid so that
rank 0 → row 0, rank 7 → row 7.  The visual orientation (White at the
bottom) matches the standard chessboard layout.
"""

import chess
import numpy as np


# ---------------------------------------------------------------------------
# Plane index definitions — single source of truth
# ---------------------------------------------------------------------------

# (colour, piece_type) → plane index in the output tensor
_PLANE_INDEX: dict[tuple[bool, int], int] = {
    (chess.WHITE, chess.PAWN):   0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK):   3,
    (chess.WHITE, chess.QUEEN):  4,
    (chess.WHITE, chess.KING):   5,
    (chess.BLACK, chess.PAWN):   6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK):   9,
    (chess.BLACK, chess.QUEEN):  10,
    (chess.BLACK, chess.KING):   11,
}

# Human-readable label for each plane — useful for debugging
PLANE_LABELS: list[str] = [
    "White Pawns",   "White Knights", "White Bishops",
    "White Rooks",   "White Queens",  "White King",
    "Black Pawns",   "Black Knights", "Black Bishops",
    "Black Rooks",   "Black Queens",  "Black King",
    "Turn (1=White, 0=Black)",
]

NUM_PLANES      = 13
TURN_PLANE      = 12   # index of the turn indicator plane
NUM_PIECE_PLANES = 12  # planes 0–11 are piece planes
BOARD_SIZE  = 8


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encode a chess position as a (13, 8, 8) float32 NumPy array.

    Planes 0–11 are binary 8×8 grids showing piece positions.
    Plane 12 is a turn indicator: all 1.0 if White to move, all 0.0 if Black.

    Parameters
    ----------
    board : chess.Board
        Any legal chess position (mid-game, starting, custom FEN, etc.).

    Returns
    -------
    np.ndarray
        Shape (13, 8, 8), dtype float32.
        All values are either 0.0 or 1.0.

    Examples
    --------
    >>> import chess
    >>> board  = chess.Board()
    >>> tensor = encode_board(board)
    >>> tensor.shape
    (13, 8, 8)
    >>> tensor.dtype
    dtype('float32')

    # White has 8 pawns on rank 2 at the start — plane 0 should sum to 8
    >>> tensor[0].sum()
    8.0

    # It is White's turn at the start — turn plane should be all 1s
    >>> tensor[12].sum()
    64.0
    """
    tensor = np.zeros((NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # --- Planes 0–11: piece positions ---
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        plane = _PLANE_INDEX[(piece.color, piece.piece_type)]
        rank  = chess.square_rank(square)   # 0–7  (rank 1 → 0, rank 8 → 7)
        file  = chess.square_file(square)   # 0–7  (a-file → 0, h-file → 7)

        tensor[plane, rank, file] = 1.0

    # --- Plane 12: turn indicator ---
    # Fill the entire plane with 1.0 for White's turn, leave as 0.0 for Black.
    if board.turn == chess.WHITE:
        tensor[TURN_PLANE, :, :] = 1.0

    return tensor


def decode_plane(tensor: np.ndarray, plane: int) -> str:
    """
    Return a human-readable ASCII grid for a single plane.

    Useful for debugging — prints 1 where a piece exists, . elsewhere.

    Parameters
    ----------
    tensor : np.ndarray
        Output of encode_board() — shape (12, 8, 8).
    plane : int
        Index 0–12. Planes 0–11 are piece types; plane 12 is the turn indicator.

    Returns
    -------
    str
        8-line ASCII representation, rank 8 at top (standard orientation).

    Examples
    --------
    >>> print(decode_plane(encode_board(chess.Board()), 0))
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    1 1 1 1 1 1 1 1
    . . . . . . . .
    """
    if not (0 <= plane < NUM_PLANES):
        raise ValueError(f"plane must be 0\u2013{NUM_PLANES - 1}, got {plane}")

    label = PLANE_LABELS[plane]
    grid  = tensor[plane]

    # Flip vertically so rank 8 appears at the top (standard board view)
    rows = []
    for rank in range(BOARD_SIZE - 1, -1, -1):
        row = " ".join("1" if grid[rank, file] else "." for file in range(BOARD_SIZE))
        rows.append(row)

    header = f"Plane {plane:>2} — {label}"
    border = "-" * len(header)
    return f"{header}\n{border}\n" + "\n".join(rows)


def encode_board_summary(board: chess.Board) -> None:
    """
    Print a full summary of all 12 planes for a given position.

    Intended for development and debugging only.

    Parameters
    ----------
    board : chess.Board
        Position to inspect.
    """
    tensor = encode_board(board)
    print(f"Board tensor shape : {tensor.shape}")
    print(f"Board tensor dtype : {tensor.dtype}")
    print(f"Non-zero cells     : {int(tensor.sum())} (expected 96 at start: 32 pieces + 64 turn cells)\n")
    for plane in range(NUM_PLANES):
        print(decode_plane(tensor, plane))
        print()