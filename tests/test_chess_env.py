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
        # Make a legal move first so the e2 pawn is no longer at its starting rank
        env.step(chess.Move.from_uci("e2e4"))
        env.step(chess.Move.from_uci("e7e5"))
        # Now trying to move a pawn illegally (a2a5 is a two-square jump)
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

        We will test the move sequence 1. e4 d5 2. exd5, where White's
        second move is a capture.
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