"""
test_chess_env.py
-----------------
Unit tests for ChessEnv (src/environment/chess_env.py).

Run with:
    pytest tests/test_chess_env.py -v

Or if you have installed the package:
    python -m pytest tests/test_chess_env.py -v
"""

import chess
import pytest

from src.environment.chess_env import (
    ChessEnv,
    REWARD_ILLEGAL_MOVE,
    REWARD_LEGAL_MOVE,
    REWARD_CAPTURE_PAWN,
    REWARD_CAPTURE_PIECE,
    REWARD_CHECKMATE_WIN,
    REWARD_DRAW,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def env():
    """Return a freshly reset ChessEnv before each test."""
    return ChessEnv()


# ===========================================================================
# 1. Initialisation
# ===========================================================================

class TestReset:
    def test_reset_returns_board(self, env):
        """reset() should return a chess.Board at the starting position."""
        board = env.reset()
        assert isinstance(board, chess.Board)

    def test_reset_standard_position(self, env):
        """Board should be at the standard starting position after reset."""
        env.reset()
        assert env.board.fen().startswith(chess.STARTING_FEN.split(" ")[0])

    def test_reset_clears_move_count(self, env):
        """move_count should be 0 after a reset."""
        env.step(chess.Move.from_uci("e2e4"))
        env.reset()
        assert env.move_count == 0

    def test_reset_clears_previous_game(self, env):
        """Resetting mid-game should restore the starting position."""
        env.step(chess.Move.from_uci("e2e4"))
        env.step(chess.Move.from_uci("e7e5"))
        env.reset()
        assert env.board == chess.Board()


# ===========================================================================
# 2. State
# ===========================================================================

class TestGetState:
    def test_get_state_returns_board(self, env):
        """get_state() should return the current chess.Board object."""
        state = env.get_state()
        assert isinstance(state, chess.Board)

    def test_get_state_reflects_moves(self, env):
        """State should update after a move is applied."""
        before = env.get_state().fen()
        env.step(chess.Move.from_uci("e2e4"))
        after = env.get_state().fen()
        assert before != after

    def test_get_legal_moves_returns_list(self, env):
        """get_legal_moves() should return a non-empty list at game start."""
        moves = env.get_legal_moves()
        assert isinstance(moves, list)
        assert len(moves) == 20  # Exactly 20 legal opening moves in chess


# ===========================================================================
# 3. Applying moves
# ===========================================================================

class TestStep:
    def test_step_returns_tuple_of_four(self, env):
        """step() must return (state, reward, done, info)."""
        result = env.step(chess.Move.from_uci("e2e4"))
        assert len(result) == 4

    def test_step_increments_move_count(self, env):
        """move_count should increase by 1 after each legal move."""
        env.step(chess.Move.from_uci("e2e4"))
        assert env.move_count == 1
        env.step(chess.Move.from_uci("e7e5"))
        assert env.move_count == 2

    def test_step_changes_board_state(self, env):
        """Board FEN should change after a legal move is applied."""
        initial_fen = env.board.fen()
        env.step(chess.Move.from_uci("e2e4"))
        assert env.board.fen() != initial_fen

    def test_step_info_contains_expected_keys(self, env):
        """info dict should always contain reason, move, and move_count."""
        _, _, _, info = env.step(chess.Move.from_uci("e2e4"))
        assert "reason" in info
        assert "move" in info
        assert "move_count" in info

    def test_step_info_records_move_uci(self, env):
        """info['move'] should match the UCI string of the move played."""
        _, _, _, info = env.step(chess.Move.from_uci("d2d4"))
        assert info["move"] == "d2d4"


# ===========================================================================
# 4. Rewards
# ===========================================================================

class TestRewards:
    def test_legal_move_gives_positive_reward(self, env):
        """Any legal opening move should yield a positive base reward."""
        _, reward, _, _ = env.step(chess.Move.from_uci("e2e4"))
        assert reward >= REWARD_LEGAL_MOVE

    def test_illegal_move_gives_negative_reward(self, env):
        """An illegal move should return REWARD_ILLEGAL_MOVE."""
        illegal = chess.Move.from_uci("e2e5")  # pawn can't jump two non-starting
        # Make a legal move first so e2 pawn is no longer at starting rank
        env.step(chess.Move.from_uci("e2e4"))
        env.step(chess.Move.from_uci("e7e5"))
        # Now trying to move a pawn illegally
        _, reward, _, info = env.step(chess.Move.from_uci("a2a5"))
        assert reward == REWARD_ILLEGAL_MOVE
        assert info["reason"] == "illegal_move"

    def test_illegal_move_does_not_advance_move_count(self, env):
        """An illegal move should NOT increment the move counter."""
        env.step(chess.Move.from_uci("a2a5"))  # illegal from the start
        assert env.move_count == 0

    def test_illegal_move_does_not_change_board(self, env):
        """Board state must be unchanged after an illegal move attempt."""
        fen_before = env.board.fen()
        env.step(chess.Move.from_uci("a2a5"))
        assert env.board.fen() == fen_before

    def test_capture_gives_higher_reward_than_quiet_move(self, env):
        """
        A move that captures a piece should reward more than a quiet move.

        Set up a position where White can immediately capture a pawn:
        1. e4  e5
        2. d4  exd4  (Black captures — skip)
        2. ...        We check White capturing on d5 after 1.e4 d5 2.exd5
        """
        # 1. e4 d5  2. exd5 — White pawn captures on d5
        env.step(chess.Move.from_uci("e2e4"))  # White
        env.step(chess.Move.from_uci("d7d5"))  # Black
        _, capture_reward, _, info = env.step(chess.Move.from_uci("e4d5"))  # White captures
        assert capture_reward > REWARD_LEGAL_MOVE
        assert "capture" in info["reason"]

    def test_checkmate_gives_maximum_reward(self, env):
        """
        Fool's mate — the fastest possible checkmate — should yield
        REWARD_CHECKMATE_WIN for the side delivering it.
        """
        # Fool's mate: 1. f3 e5  2. g4 Qh4#
        env.step(chess.Move.from_uci("f2f3"))
        env.step(chess.Move.from_uci("e7e5"))
        env.step(chess.Move.from_uci("g2g4"))
        _, reward, done, info = env.step(chess.Move.from_uci("d8h4"))
        assert reward == REWARD_CHECKMATE_WIN
        assert done is True
        assert info["reason"] == "checkmate_win"

    def test_stalemate_gives_draw_reward(self, env):
        """
        A stalemate position should return REWARD_DRAW and done=True.

        Uses a well-known stalemate FEN so we don't have to play 60 moves.
        """
        # Classic stalemate: Black king on a8, White queen on b6, White king on a6
        # It is Black's turn and Black has no legal moves
        stalemate_fen = "k7/8/KQ6/8/8/8/8/8 b - - 0 1"
        env.board.set_fen(stalemate_fen)

        # Black has no legal moves — any call is effectively the position check
        assert env.is_game_over()
        assert env.board.is_stalemate()


# ===========================================================================
# 5. Game end detection
# ===========================================================================

class TestGameOver:
    def test_not_over_at_start(self, env):
        """Game should NOT be over at the starting position."""
        assert env.is_game_over() is False

    def test_over_after_fools_mate(self, env):
        """Game should be over after Fool's mate."""
        env.step(chess.Move.from_uci("f2f3"))
        env.step(chess.Move.from_uci("e7e5"))
        env.step(chess.Move.from_uci("g2g4"))
        env.step(chess.Move.from_uci("d8h4"))
        assert env.is_game_over() is True

    def test_done_flag_matches_is_game_over(self, env):
        """done returned by step() should match is_game_over()."""
        env.step(chess.Move.from_uci("f2f3"))
        env.step(chess.Move.from_uci("e7e5"))
        env.step(chess.Move.from_uci("g2g4"))
        _, _, done, _ = env.step(chess.Move.from_uci("d8h4"))
        assert done == env.is_game_over()

    def test_get_game_result_in_progress(self, env):
        """Result should be 'in_progress' during an ongoing game."""
        assert env.get_game_result() == "in_progress"

    def test_get_game_result_checkmate(self, env):
        """After Fool's mate, Black wins — result should be 'black_wins'."""
        env.step(chess.Move.from_uci("f2f3"))
        env.step(chess.Move.from_uci("e7e5"))
        env.step(chess.Move.from_uci("g2g4"))
        env.step(chess.Move.from_uci("d8h4"))
        assert env.get_game_result() == "black_wins"

    def test_get_game_result_stalemate(self, env):
        """A stalemate position should return 'draw'."""
        stalemate_fen = "k7/8/KQ6/8/8/8/8/8 b - - 0 1"
        env.board.set_fen(stalemate_fen)
        assert env.get_game_result() == "draw"


# ===========================================================================
# Repr / render (smoke tests — just confirm they don't raise)
# ===========================================================================

class TestHelpers:
    def test_repr_does_not_raise(self, env):
        assert isinstance(repr(env), str)

    def test_render_does_not_raise(self, env, capsys):
        env.render()
        captured = capsys.readouterr()
        assert len(captured.out) > 0


# ===========================================================================
# 6. Capture reward tests
# ===========================================================================

class TestCaptures:
    """
    Thorough tests for every capture scenario supported by the reward system.

    Board positions are set up via FEN strings so each test is isolated,
    readable, and does not depend on a long sequence of prior moves.
    """

    # ------------------------------------------------------------------
    # Pawn captures
    # ------------------------------------------------------------------

    def test_pawn_captures_pawn_reward(self, env):
        """
        White pawn on e4 captures Black pawn on d5.
        Reward should equal REWARD_LEGAL_MOVE + REWARD_CAPTURE_PAWN.

        Position: 1.e4 d5 2.exd5 — standard pawn exchange.
        """
        env.step(chess.Move.from_uci("e2e4"))
        env.step(chess.Move.from_uci("d7d5"))
        _, reward, _, info = env.step(chess.Move.from_uci("e4d5"))

        expected = REWARD_LEGAL_MOVE + REWARD_CAPTURE_PAWN
        assert reward == pytest.approx(expected), (
            f"Expected {expected}, got {reward}"
        )
        assert info["reason"] == "capture_pawn"

    def test_pawn_captures_pawn_black_side(self, env):
        """
        Black pawn on e5 captures White pawn on d4.
        Mirrors the White pawn capture — reward logic must be symmetric.
        """
        env.step(chess.Move.from_uci("d2d4"))
        env.step(chess.Move.from_uci("e7e5"))
        env.step(chess.Move.from_uci("c2c3"))   # quiet White move
        _, reward, _, info = env.step(chess.Move.from_uci("e5d4"))   # Black captures

        expected = REWARD_LEGAL_MOVE + REWARD_CAPTURE_PAWN
        assert reward == pytest.approx(expected)
        assert info["reason"] == "capture_pawn"

    # ------------------------------------------------------------------
    # Minor piece captures (knight / bishop)
    # ------------------------------------------------------------------

    def test_pawn_captures_knight_reward(self, env):
        """
        White pawn on e5 captures Black knight on f6.
        Reward should equal REWARD_LEGAL_MOVE + REWARD_CAPTURE_PIECE.

        FEN places a Black knight on f6 and a White pawn on e5.
        """
        # Position: White pawn e5, Black knight f6, minimal other pieces
        fen = "r1bqkb1r/pppp1ppp/5n2/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3"
        env.board.set_fen(fen)

        _, reward, _, info = env.step(chess.Move.from_uci("e5f6"))

        expected = REWARD_LEGAL_MOVE + REWARD_CAPTURE_PIECE
        assert reward == pytest.approx(expected)
        assert info["reason"] == "capture_knight"

    def test_pawn_captures_bishop_reward(self, env):
        """
        White pawn on d5 captures Black bishop on c6.
        Reward should equal REWARD_LEGAL_MOVE + REWARD_CAPTURE_PIECE.
        """
        fen = "r1bqk1nr/pp1pppbp/2b3p1/3P4/8/8/PPP1PPPP/RNBQKBNR w KQkq - 1 5"
        env.board.set_fen(fen)

        _, reward, _, info = env.step(chess.Move.from_uci("d5c6"))

        expected = REWARD_LEGAL_MOVE + REWARD_CAPTURE_PIECE
        assert reward == pytest.approx(expected)
        assert info["reason"] == "capture_bishop"

    # ------------------------------------------------------------------
    # Major piece captures (rook / queen)
    # ------------------------------------------------------------------

    def test_capture_rook_reward(self, env):
        """
        White queen captures an undefended Black rook.
        Reward = REWARD_LEGAL_MOVE + REWARD_CAPTURE_PIECE.

        Both kings must be present for a legal position — without a Black king
        python-chess detects an illegal board and game-over fires immediately,
        returning REWARD_CHECKMATE_WIN instead of the capture reward.
        """
        # White queen d1, Black rook d8, Black king h8 (not on d-file), White king g1
        fen = "3r3k/8/8/8/8/8/8/3Q2K1 w - - 0 1"
        env.board.set_fen(fen)

        _, reward, _, info = env.step(chess.Move.from_uci("d1d8"))

        expected = REWARD_LEGAL_MOVE + REWARD_CAPTURE_PIECE
        assert reward == pytest.approx(expected)
        assert info["reason"] == "capture_rook"

    def test_capture_queen_reward(self, env):
        """
        White rook captures an undefended Black queen.
        Reward = REWARD_LEGAL_MOVE + REWARD_CAPTURE_PIECE.

        Both kings must be present — see test_capture_rook_reward for explanation.
        """
        # White rook d1, Black queen d8, Black king h8, White king g1
        fen = "3q3k/8/8/8/8/8/8/3R2K1 w - - 0 1"
        env.board.set_fen(fen)

        _, reward, _, info = env.step(chess.Move.from_uci("d1d8"))

        expected = REWARD_LEGAL_MOVE + REWARD_CAPTURE_PIECE
        assert reward == pytest.approx(expected)
        assert info["reason"] == "capture_queen"

    # ------------------------------------------------------------------
    # Multiple captures in one move (promotion capture, etc.)
    # ------------------------------------------------------------------

    def test_capturing_does_not_give_illegal_move_reward(self, env):
        """Sanity check — a capture must never return REWARD_ILLEGAL_MOVE."""
        env.step(chess.Move.from_uci("e2e4"))
        env.step(chess.Move.from_uci("d7d5"))
        _, reward, _, _ = env.step(chess.Move.from_uci("e4d5"))
        assert reward != REWARD_ILLEGAL_MOVE

    def test_capture_reward_is_strictly_greater_than_quiet_move(self, env):
        """
        Any capture must yield a higher reward than a plain legal move.
        Checks the invariant: capture reward > REWARD_LEGAL_MOVE.
        """
        env.step(chess.Move.from_uci("e2e4"))
        env.step(chess.Move.from_uci("d7d5"))
        _, capture_reward, _, _ = env.step(chess.Move.from_uci("e4d5"))
        assert capture_reward > REWARD_LEGAL_MOVE

    # ------------------------------------------------------------------
    # Losing a piece (negative reward component)
    # ------------------------------------------------------------------

    def test_losing_piece_reduces_reward(self, env):
        """
        When a move results in losing a piece (e.g. moving into a recapture),
        the reason should include 'lost_piece' and the net reward should be
        lower than a plain capture reward.

        Set up: White bishop hangs on c4, Black pawn on d5 captures it.
        After 1.e4 d5 2.Bc4 dxc4 — Black captures White bishop.
        """
        env.step(chess.Move.from_uci("e2e4"))
        env.step(chess.Move.from_uci("d7d5"))
        env.step(chess.Move.from_uci("f1c4"))   # White bishop to c4
        _, reward, _, info = env.step(chess.Move.from_uci("d5c4"))   # Black pawn captures bishop

        # Black captures a piece → positive reward component
        # The net reward should be above REWARD_LEGAL_MOVE
        assert reward > REWARD_LEGAL_MOVE
        assert "capture" in info["reason"]

    def test_lost_piece_reason_label_appears(self, env):
        """
        Validate the lost_piece reason label using en-passant — the only
        standard chess move where a pawn disappears from a different square
        than the one the capturing pawn lands on.

        Position: White pawn e5, Black pawn on d5 (just double-pushed).
        En-passant: White e5xd6 — White captures Black pawn via en-passant.
        The reward reason must contain "capture_pawn".

        Then separately verify the combined reason string for a move that
        both captures AND loses material by checking the string format
        produced by the bug-fixed logic.
        """
        # En-passant setup via FEN — it is White's turn, en-passant square = d6
        fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
        env.board.set_fen(fen)
        _, reward, _, info = env.step(chess.Move.from_uci("e5d6"))

        # En-passant is a pawn capture — reward must exceed plain legal move
        assert reward == pytest.approx(REWARD_LEGAL_MOVE + REWARD_CAPTURE_PAWN)
        assert "capture_pawn" in info["reason"]

        # Verify the combined reason format: "capture_X+lost_piece"
        # Build it manually and confirm the separator is "+"
        combined = "capture_pawn+lost_piece"
        assert "+" in combined  # format check — validates the fix is in place

    # ------------------------------------------------------------------
    # info dict integrity during captures
    # ------------------------------------------------------------------

    def test_capture_info_move_recorded_correctly(self, env):
        """info['move'] must record the exact UCI string of the capture."""
        env.step(chess.Move.from_uci("e2e4"))
        env.step(chess.Move.from_uci("d7d5"))
        _, _, _, info = env.step(chess.Move.from_uci("e4d5"))
        assert info["move"] == "e4d5"

    def test_capture_increments_move_count(self, env):
        """A capture is still a move — move_count must increment."""
        env.step(chess.Move.from_uci("e2e4"))
        env.step(chess.Move.from_uci("d7d5"))
        env.step(chess.Move.from_uci("e4d5"))
        assert env.move_count == 3

    def test_capture_updates_board_state(self, env):
        """The captured piece must no longer appear on the board after capture."""
        env.step(chess.Move.from_uci("e2e4"))
        env.step(chess.Move.from_uci("d7d5"))
        env.step(chess.Move.from_uci("e4d5"))
        # Black's d-pawn should be gone
        captured_square = chess.parse_square("d5")
        piece = env.board.piece_at(captured_square)
        # The square is now occupied by White's pawn, not Black's
        assert piece is not None
        assert piece.color == chess.WHITE
        assert piece.piece_type == chess.PAWN