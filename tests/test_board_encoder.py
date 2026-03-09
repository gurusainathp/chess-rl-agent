"""
test_board_encoder.py
---------------------
Unit tests for board_encoder.py.

Run with:
    pytest tests/test_board_encoder.py -v
"""

import chess
import numpy as np
import pytest

from src.environment.board_encoder import (
    encode_board,
    decode_plane,
    PLANE_LABELS,
    NUM_PLANES,
    NUM_PIECE_PLANES,
    TURN_PLANE,
    BOARD_SIZE,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def starting_tensor():
    """Encoded tensor of the standard starting position."""
    return encode_board(chess.Board())


@pytest.fixture
def empty_board_tensor():
    """Encoded tensor of a completely empty board."""
    return encode_board(chess.Board(fen=None))


# ===========================================================================
# 1. Output shape and dtype
# ===========================================================================

class TestOutputShape:
    def test_shape_is_13_8_8(self, starting_tensor):
        """Tensor must be exactly (13, 8, 8) — 12 piece planes + 1 turn plane."""
        assert starting_tensor.shape == (13, 8, 8)

    def test_dtype_is_float32(self, starting_tensor):
        """Tensor must use float32 for PyTorch compatibility."""
        assert starting_tensor.dtype == np.float32

    def test_shape_after_moves(self):
        """Shape must remain (12, 8, 8) regardless of position."""
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        board.push(chess.Move.from_uci("e7e5"))
        assert encode_board(board).shape == (13, 8, 8)

    def test_shape_from_custom_fen(self):
        """Shape must remain (12, 8, 8) for any arbitrary FEN."""
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        assert encode_board(chess.Board(fen)).shape == (13, 8, 8)


# ===========================================================================
# 2. Binary values
# ===========================================================================

class TestBinaryValues:
    def test_only_zeros_and_ones(self, starting_tensor):
        """Every cell must be exactly 0.0 or 1.0 — nothing else."""
        unique = np.unique(starting_tensor)
        assert set(unique).issubset({0.0, 1.0})

    def test_empty_board_piece_planes_are_all_zeros(self, empty_board_tensor):
        """
        An empty board has no pieces — piece planes 0–11 must be all 0.
        The turn plane (12) is NOT checked here because python-chess still
        reports board.turn == WHITE on an empty board, so plane 12 = all 1s.
        """
        assert np.all(empty_board_tensor[:NUM_PIECE_PLANES] == 0.0)

    def test_no_plane_exceeds_starting_piece_count(self, starting_tensor):
        """
        At the start, no piece-type plane sums to more than 8.
        Pawn planes = 8, king/queen planes = 1.
        The turn plane (12) is excluded — it legitimately sums to 64.
        """
        for plane in range(NUM_PIECE_PLANES):   # 0–11 only, skip turn plane
            assert starting_tensor[plane].sum() <= 8


# ===========================================================================
# 3. Piece counts at the starting position
# ===========================================================================

class TestStartingPieceCounts:
    """
    At the standard starting position each side has:
        8 pawns, 2 knights, 2 bishops, 2 rooks, 1 queen, 1 king
    Total occupied squares = 32.
    """

    def test_total_sum_is_96_at_start(self, starting_tensor):
        """32 piece cells + 64 turn-plane cells (White to move) = 96."""
        assert starting_tensor.sum() == 96.0

    # White pieces (planes 0–5)
    def test_white_pawns_count_8(self, starting_tensor):
        assert starting_tensor[0].sum() == 8.0

    def test_white_knights_count_2(self, starting_tensor):
        assert starting_tensor[1].sum() == 2.0

    def test_white_bishops_count_2(self, starting_tensor):
        assert starting_tensor[2].sum() == 2.0

    def test_white_rooks_count_2(self, starting_tensor):
        assert starting_tensor[3].sum() == 2.0

    def test_white_queens_count_1(self, starting_tensor):
        assert starting_tensor[4].sum() == 1.0

    def test_white_king_count_1(self, starting_tensor):
        assert starting_tensor[5].sum() == 1.0

    # Black pieces (planes 6–11)
    def test_black_pawns_count_8(self, starting_tensor):
        assert starting_tensor[6].sum() == 8.0

    def test_black_knights_count_2(self, starting_tensor):
        assert starting_tensor[7].sum() == 2.0

    def test_black_bishops_count_2(self, starting_tensor):
        assert starting_tensor[8].sum() == 2.0

    def test_black_rooks_count_2(self, starting_tensor):
        assert starting_tensor[9].sum() == 2.0

    def test_black_queens_count_1(self, starting_tensor):
        assert starting_tensor[10].sum() == 1.0

    def test_black_king_count_1(self, starting_tensor):
        assert starting_tensor[11].sum() == 1.0


# ===========================================================================
# 4. Correct square positions
# ===========================================================================

class TestSquarePositions:
    """
    Verify that specific pieces appear at the exact (rank, file) cell
    we expect, using known starting-position squares.

    python-chess ranks: 0 = rank 1, 7 = rank 8
    python-chess files: 0 = a-file, 7 = h-file
    """

    def test_white_pawns_on_rank_1(self, starting_tensor):
        """White pawns start on rank 2 → index 1 in the array."""
        pawn_plane = starting_tensor[0]   # plane 0 = white pawns
        assert np.all(pawn_plane[1, :] == 1.0), "All 8 white pawns should be on rank-index 1"
        assert pawn_plane[0, :].sum() == 0.0,   "No white pawns on rank-index 0"

    def test_black_pawns_on_rank_6(self, starting_tensor):
        """Black pawns start on rank 7 → index 6 in the array."""
        pawn_plane = starting_tensor[6]   # plane 6 = black pawns
        assert np.all(pawn_plane[6, :] == 1.0), "All 8 black pawns should be on rank-index 6"
        assert pawn_plane[7, :].sum() == 0.0,   "No black pawns on rank-index 7"

    def test_white_king_on_e1(self, starting_tensor):
        """White king starts on e1 → rank 0, file 4."""
        king_plane = starting_tensor[5]   # plane 5 = white king
        assert king_plane[0, 4] == 1.0

    def test_black_king_on_e8(self, starting_tensor):
        """Black king starts on e8 → rank 7, file 4."""
        king_plane = starting_tensor[11]  # plane 11 = black king
        assert king_plane[7, 4] == 1.0

    def test_white_queen_on_d1(self, starting_tensor):
        """White queen starts on d1 → rank 0, file 3."""
        queen_plane = starting_tensor[4]  # plane 4 = white queens
        assert queen_plane[0, 3] == 1.0

    def test_white_rooks_on_a1_and_h1(self, starting_tensor):
        """White rooks start on a1 (rank 0, file 0) and h1 (rank 0, file 7)."""
        rook_plane = starting_tensor[3]   # plane 3 = white rooks
        assert rook_plane[0, 0] == 1.0
        assert rook_plane[0, 7] == 1.0


# ===========================================================================
# 5. Tensor updates correctly after moves
# ===========================================================================

class TestDynamicUpdates:
    def test_pawn_moves_to_new_square(self):
        """
        After 1. e4, White's e-pawn should be on e4 (rank 3, file 4)
        and NOT on e2 (rank 1, file 4) anymore.
        """
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        tensor = encode_board(board)
        pawn_plane = tensor[0]             # white pawns

        assert pawn_plane[3, 4] == 1.0,   "e-pawn should now be on rank 3 (e4)"
        assert pawn_plane[1, 4] == 0.0,   "e-pawn should no longer be on rank 1 (e2)"

    def test_captured_piece_disappears(self):
        """
        After 1. e4 d5  2. exd5, Black's d-pawn (rank 4, file 3) is captured.
        It should no longer appear in the black pawn plane.
        """
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        board.push(chess.Move.from_uci("d7d5"))
        board.push(chess.Move.from_uci("e4d5"))   # White captures
        tensor = encode_board(board)

        black_pawn_plane = tensor[6]
        assert black_pawn_plane[4, 3] == 0.0, "Captured pawn should not appear"

    def test_total_pieces_decreases_after_capture(self):
        """
        After a capture the total number of 1.0 cells should decrease by 1.
        """
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        board.push(chess.Move.from_uci("d7d5"))
        board.push(chess.Move.from_uci("e4d5"))
        tensor = encode_board(board)
        # After e4xd5 it is Black's turn → turn plane = all 0s
        # Total = 31 piece cells + 0 turn cells = 31
        assert tensor.sum() == 31.0, "31 pieces remain; turn plane is 0 (Black to move)"

    def test_no_plane_overlap(self):
        """
        Two pieces can never share the same square — summing all planes
        cell-wise must produce only 0s and 1s (no cell can be > 1).
        """
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        board.push(chess.Move.from_uci("e7e5"))
        tensor = encode_board(board)
        # Only check piece planes 0–11; the turn plane is allowed to be all 1s
        collapsed = tensor[:NUM_PIECE_PLANES].sum(axis=0)
        assert np.all(collapsed <= 1.0), "No two pieces can occupy the same square"



# ===========================================================================
# 6. Turn plane (plane 12)
# ===========================================================================

class TestTurnPlane:
    def test_turn_plane_all_ones_when_white_to_move(self, starting_tensor):
        """At the start it is White\'s turn — plane 12 must be all 1.0."""
        assert starting_tensor[TURN_PLANE].sum() == 64.0
        assert np.all(starting_tensor[TURN_PLANE] == 1.0)

    def test_turn_plane_all_zeros_when_black_to_move(self):
        """After 1. e4 it is Black\'s turn — plane 12 must be all 0.0."""
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        tensor = encode_board(board)
        assert tensor[TURN_PLANE].sum() == 0.0
        assert np.all(tensor[TURN_PLANE] == 0.0)

    def test_turn_plane_flips_every_move(self):
        """The turn plane should alternate 1 → 0 → 1 with each move."""
        board = chess.Board()
        # After White moves → Black to move → plane = 0.0
        # After Black moves → White to move → plane = 1.0
        moves    = ["e2e4", "e7e5", "d2d4", "d7d5"]
        expected = [0.0,    1.0,    0.0,    1.0   ]
        for uci, exp in zip(moves, expected):
            board.push(chess.Move.from_uci(uci))
            tensor = encode_board(board)
            actual = tensor[TURN_PLANE].mean()   # mean of 64 cells: 1.0=White, 0.0=Black
            assert actual == exp, (
                f"After {uci}: expected turn plane mean={exp}, got {actual}"
            )

    def test_turn_plane_does_not_affect_piece_planes(self):
        """The turn plane must never interfere with piece position data."""
        board = chess.Board()
        tensor = encode_board(board)
        # Piece planes 0–11 should still sum to 32 (piece positions only)
        assert tensor[:NUM_PIECE_PLANES].sum() == 32.0

    def test_turn_plane_is_uniform(self):
        """All 64 cells of the turn plane must be identical (all 0 or all 1)."""
        for uci in ["e2e4", "e7e5"]:
            board = chess.Board()
            board.push(chess.Move.from_uci(uci))
            tensor = encode_board(board)
            unique_values = np.unique(tensor[TURN_PLANE])
            assert len(unique_values) == 1, "Turn plane must be uniformly 0 or 1"


# ===========================================================================
# 7. decode_plane helper
# ===========================================================================

class TestDecodePlane:
    def test_returns_string(self, starting_tensor):
        result = decode_plane(starting_tensor, 0)
        assert isinstance(result, str)

    def test_contains_plane_label(self, starting_tensor):
        result = decode_plane(starting_tensor, 0)
        assert "White Pawns" in result

    def test_invalid_plane_raises(self, starting_tensor):
        with pytest.raises(ValueError):
            decode_plane(starting_tensor, 13)  # one beyond the valid range (0–12)

    def test_negative_plane_raises(self, starting_tensor):
        with pytest.raises(ValueError):
            decode_plane(starting_tensor, -1)

    def test_all_planes_renderable(self, starting_tensor):
        """decode_plane should not raise for any valid plane index."""
        for plane in range(NUM_PLANES):
            result = decode_plane(starting_tensor, plane)
            assert len(result) > 0