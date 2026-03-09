"""
test_evaluation.py
------------------
Unit tests for random_agent.py and evaluate_model.py.

Run with:
    pytest tests/test_evaluation.py -v
"""

import json
import os
import chess
import pytest
import torch

from src.evaluation.random_agent import RandomAgent
from src.evaluation.evaluate_model import (
    EvaluationConfig,
    EvaluationResult,
    GameResult,
    evaluate,
    _play_one_game,
)
from src.models.policy_network import PolicyNetwork


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    m = PolicyNetwork()
    m.eval()
    return m


@pytest.fixture
def agent():
    return RandomAgent(seed=0)


# ===========================================================================
# 1. RandomAgent
# ===========================================================================

class TestRandomAgent:
    def test_select_move_returns_chess_move(self, agent):
        board = chess.Board()
        move  = agent.select_move(board)
        assert isinstance(move, chess.Move)

    def test_selected_move_is_legal(self, agent):
        board = chess.Board()
        move  = agent.select_move(board)
        assert move in board.legal_moves

    def test_selected_move_is_legal_midgame(self, agent):
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        move  = agent.select_move(board)
        assert move in board.legal_moves

    def test_raises_on_game_over(self, agent):
        """select_move must raise ValueError when there are no legal moves."""
        board = chess.Board("8/8/8/8/8/8/8/k1K5 w - - 0 1")
        # Force a stalemate/checkmate position by using a known terminal FEN
        # Simplest: scholar's mate — black is checkmated
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
        board.push_san("Qxf7#")
        assert board.is_game_over()
        with pytest.raises(ValueError, match="No legal moves"):
            agent.select_move(board)

    def test_seeded_agent_is_deterministic(self):
        """Same seed + same board must always produce the same move."""
        board = chess.Board()
        a1    = RandomAgent(seed=99)
        a2    = RandomAgent(seed=99)
        assert a1.select_move(board) == a2.select_move(board)

    def test_different_seeds_may_differ(self):
        """Different seeds should (very likely) produce different moves over
        multiple calls — not guaranteed every single time, so we try 20."""
        board  = chess.Board()
        moves1 = [RandomAgent(seed=i).select_move(board) for i in range(20)]
        moves2 = [RandomAgent(seed=i + 100).select_move(board) for i in range(20)]
        assert moves1 != moves2

    def test_reset_restores_sequence(self):
        """After reset(), the agent replays the same sequence of choices."""
        agent = RandomAgent(seed=7)
        board = chess.Board()
        first_run = [agent.select_move(chess.Board()) for _ in range(5)]
        agent.reset()
        second_run = [agent.select_move(chess.Board()) for _ in range(5)]
        assert first_run == second_run

    def test_none_seed_does_not_crash(self):
        """seed=None (default) must work without errors."""
        agent = RandomAgent(seed=None)
        move  = agent.select_move(chess.Board())
        assert isinstance(move, chess.Move)

    def test_repr_contains_seed(self):
        assert "42" in repr(RandomAgent(seed=42))


# ===========================================================================
# 2. EvaluationConfig
# ===========================================================================

class TestEvaluationConfig:
    def test_defaults(self):
        cfg = EvaluationConfig()
        assert cfg.n_games         == 100
        assert cfg.max_moves       == 200
        assert cfg.temperature     == 0.1
        assert cfg.alternate_sides is True
        assert cfg.save_path       is None
        assert cfg.verbose         is False
        assert cfg.device          == "cpu"

    def test_custom_values(self):
        cfg = EvaluationConfig(n_games=10, temperature=0.5, alternate_sides=False)
        assert cfg.n_games         == 10
        assert cfg.temperature     == 0.5
        assert cfg.alternate_sides is False


# ===========================================================================
# 3. GameResult
# ===========================================================================

class TestGameResult:
    def test_fields_stored_correctly(self):
        gr = GameResult(game_number=1, model_colour="white", outcome="win", n_moves=40)
        assert gr.game_number  == 1
        assert gr.model_colour == "white"
        assert gr.outcome      == "win"
        assert gr.n_moves      == 40


# ===========================================================================
# 4. EvaluationResult
# ===========================================================================

class TestEvaluationResult:
    def _make_result(self, wins=6, losses=2, draws=1, max_mv=1, avg=50.0):
        return EvaluationResult(
            total_games=wins + losses + draws + max_mv,
            wins=wins, losses=losses, draws=draws,
            max_moves_games=max_mv,
            average_game_length=avg,
        )

    def test_winrate_correct(self):
        r = self._make_result(wins=6, losses=2, draws=1, max_mv=1)
        assert pytest.approx(r.winrate) == 6 / 10

    def test_lossrate_correct(self):
        r = self._make_result(wins=6, losses=2, draws=1, max_mv=1)
        assert pytest.approx(r.lossrate) == 2 / 10

    def test_drawrate_includes_capped_games(self):
        """draws + max_moves_games both count as draws in drawrate."""
        r = self._make_result(wins=6, losses=2, draws=1, max_mv=1)
        assert pytest.approx(r.drawrate) == 2 / 10

    def test_rates_sum_to_one(self):
        r = self._make_result()
        assert pytest.approx(r.winrate + r.lossrate + r.drawrate) == 1.0

    def test_zero_games_gives_zero_rates(self):
        r = EvaluationResult(0, 0, 0, 0, 0, 0.0)
        assert r.winrate == 0.0 and r.lossrate == 0.0 and r.drawrate == 0.0

    def test_summary_is_string(self):
        assert isinstance(self._make_result().summary(), str)

    def test_summary_contains_wins(self):
        r = self._make_result(wins=6)
        assert "6" in r.summary()

    def test_to_dict_has_expected_keys(self):
        d = self._make_result().to_dict()
        for key in ("total_games", "wins", "losses", "draws", "winrate", "lossrate", "drawrate"):
            assert key in d

    def test_save_json_creates_file(self, tmp_path):
        path = str(tmp_path / "eval.json")
        self._make_result().save_json(path)
        assert os.path.exists(path)

    def test_save_json_is_valid_json(self, tmp_path):
        path = str(tmp_path / "eval.json")
        self._make_result().save_json(path)
        with open(path) as f:
            data = json.load(f)
        assert data["total_games"] == 10


# ===========================================================================
# 5. _play_one_game
# ===========================================================================

class TestPlayOneGame:
    def test_returns_valid_outcome(self, model):
        agent   = RandomAgent(seed=0)
        outcome, n_moves = _play_one_game(
            model=model, random_agent=agent,
            model_is_white=True, max_moves=200,
            temperature=0.1, device="cpu",
        )
        assert outcome in {"win", "loss", "draw", "max_moves"}

    def test_n_moves_is_positive(self, model):
        agent = RandomAgent(seed=0)
        _, n_moves = _play_one_game(
            model=model, random_agent=agent,
            model_is_white=True, max_moves=200,
            temperature=0.1, device="cpu",
        )
        assert n_moves > 0

    def test_n_moves_does_not_exceed_max(self, model):
        agent = RandomAgent(seed=0)
        _, n_moves = _play_one_game(
            model=model, random_agent=agent,
            model_is_white=True, max_moves=20,
            temperature=0.1, device="cpu",
        )
        assert n_moves <= 20

    def test_max_moves_returns_max_moves_outcome(self, model):
        """A cap of 1 must return 'max_moves' as outcome."""
        agent = RandomAgent(seed=0)
        outcome, _ = _play_one_game(
            model=model, random_agent=agent,
            model_is_white=True, max_moves=1,
            temperature=0.1, device="cpu",
        )
        assert outcome == "max_moves"

    def test_model_plays_legal_moves_only(self, model):
        """
        We push the model's moves onto a fresh board and confirm none
        raise an IllegalMoveError — all must be in board.legal_moves.
        """
        board      = chess.Board()
        agent      = RandomAgent(seed=0)
        model.eval()

        for _ in range(10):
            if board.is_game_over():
                break
            legal  = list(board.legal_moves)
            if board.turn == chess.WHITE:
                state_t = torch.tensor(
                    __import__("src.environment.board_encoder",
                               fromlist=["encode_board"]).encode_board(board),
                    dtype=torch.float32
                ).unsqueeze(0)
                with torch.no_grad():
                    move = model.select_move(state_t, legal, temperature=0.1)
            else:
                move = agent.select_move(board)
            assert move in board.legal_moves
            board.push(move)

    def test_model_as_black_returns_valid_outcome(self, model):
        agent = RandomAgent(seed=1)
        outcome, _ = _play_one_game(
            model=model, random_agent=agent,
            model_is_white=False, max_moves=200,
            temperature=0.1, device="cpu",
        )
        assert outcome in {"win", "loss", "draw", "max_moves"}


# ===========================================================================
# 6. evaluate() — aggregate behaviour
# ===========================================================================

class TestEvaluate:
    def test_returns_evaluation_result(self, model):
        cfg    = EvaluationConfig(n_games=4, max_moves=30)
        result = evaluate(model, cfg)
        assert isinstance(result, EvaluationResult)

    def test_total_games_matches_config(self, model):
        cfg    = EvaluationConfig(n_games=6, max_moves=30)
        result = evaluate(model, cfg)
        assert result.total_games == 6

    def test_counts_sum_to_total_games(self, model):
        cfg    = EvaluationConfig(n_games=8, max_moves=30)
        result = evaluate(model, cfg)
        total  = result.wins + result.losses + result.draws + result.max_moves_games
        assert total == result.total_games

    def test_game_results_list_length_matches(self, model):
        cfg    = EvaluationConfig(n_games=5, max_moves=30)
        result = evaluate(model, cfg)
        assert len(result.games) == 5

    def test_all_game_outcomes_are_valid(self, model):
        cfg    = EvaluationConfig(n_games=4, max_moves=30)
        result = evaluate(model, cfg)
        for g in result.games:
            assert g.outcome in {"win", "loss", "draw", "max_moves"}

    def test_alternating_sides_assigns_both_colours(self, model):
        """With alternate_sides=True and ≥2 games, both colours must appear."""
        cfg    = EvaluationConfig(n_games=4, max_moves=30, alternate_sides=True)
        result = evaluate(model, cfg)
        colours = {g.model_colour for g in result.games}
        assert "white" in colours and "black" in colours

    def test_no_alternation_always_white(self, model):
        """With alternate_sides=False, model always plays White."""
        cfg    = EvaluationConfig(n_games=4, max_moves=30, alternate_sides=False)
        result = evaluate(model, cfg)
        assert all(g.model_colour == "white" for g in result.games)

    def test_average_game_length_is_positive(self, model):
        cfg    = EvaluationConfig(n_games=4, max_moves=30)
        result = evaluate(model, cfg)
        assert result.average_game_length > 0.0

    def test_winrate_is_in_valid_range(self, model):
        cfg    = EvaluationConfig(n_games=4, max_moves=30)
        result = evaluate(model, cfg)
        assert 0.0 <= result.winrate <= 1.0

    def test_model_weights_unchanged_after_evaluation(self, model):
        """evaluate() must not modify model parameters."""
        params_before = {n: p.data.clone() for n, p in model.named_parameters()}
        cfg = EvaluationConfig(n_games=2, max_moves=20)
        evaluate(model, cfg)
        for name, p in model.named_parameters():
            assert torch.equal(p.data, params_before[name]), (
                f"Parameter {name} was modified by evaluate()"
            )

    def test_save_json_creates_file(self, model, tmp_path):
        path = str(tmp_path / "results.json")
        cfg  = EvaluationConfig(n_games=2, max_moves=20, save_path=path)
        evaluate(model, cfg)
        assert os.path.exists(path)

    def test_save_json_content_is_valid(self, model, tmp_path):
        path = str(tmp_path / "results.json")
        cfg  = EvaluationConfig(n_games=2, max_moves=20, save_path=path)
        evaluate(model, cfg)
        with open(path) as f:
            data = json.load(f)
        assert data["total_games"] == 2
        assert "winrate" in data

    def test_zero_games_returns_empty_result(self, model):
        cfg    = EvaluationConfig(n_games=0, max_moves=30)
        result = evaluate(model, cfg)
        assert result.total_games == 0
        assert result.winrate     == 0.0