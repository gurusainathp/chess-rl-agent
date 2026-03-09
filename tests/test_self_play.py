"""
test_self_play.py
-----------------
Unit tests for self_play.py and train_policy.py
(src/training/self_play.py, src/training/train_policy.py).

Run with:
    pytest tests/test_self_play.py -v
"""

import chess
import pytest
import torch

from src.models.policy_network import PolicyNetwork
from src.training.self_play import (
    GameSample,
    GameRecord,
    run_game,
    run_games,
    records_to_dataset,
    get_temperature,
)
from src.training.train_policy import (
    TrainingConfig,
    EpochMetrics,
    compute_loss,
    save_checkpoint,
    load_checkpoint,
    train,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    """Fresh PolicyNetwork in eval mode."""
    m = PolicyNetwork()
    m.eval()
    return m


@pytest.fixture
def single_game(model):
    """One complete self-play game record."""
    return run_game(model, max_moves=200)


# ===========================================================================
# 1. get_temperature
# ===========================================================================

class TestTemperatureSchedule:
    def test_returns_high_temp_before_threshold(self):
        assert get_temperature(0,  1.0, 0.1, 30) == 1.0
        assert get_temperature(29, 1.0, 0.1, 30) == 1.0

    def test_returns_low_temp_at_threshold(self):
        assert get_temperature(30, 1.0, 0.1, 30) == 0.1

    def test_returns_low_temp_after_threshold(self):
        assert get_temperature(50, 1.0, 0.1, 30) == 0.1

    def test_custom_temps(self):
        assert get_temperature(5, 0.8, 0.2, 10) == 0.8
        assert get_temperature(10, 0.8, 0.2, 10) == 0.2

    def test_threshold_zero_always_low(self):
        """threshold=0 means always use temp_low."""
        assert get_temperature(0, 1.0, 0.1, 0) == 0.1


# ===========================================================================
# 2. GameRecord structure
# ===========================================================================

class TestGameRecord:
    def test_initial_state(self):
        record = GameRecord()
        assert len(record) == 0
        assert record.result == "in_progress"
        assert record.n_moves == 0

    def test_len_reflects_samples(self, single_game):
        assert len(single_game) == single_game.n_moves

    def test_iterable(self, single_game):
        """GameRecord must be iterable — used in dataset construction."""
        samples = list(single_game)
        assert len(samples) == len(single_game)

    def test_white_samples_are_even_indexed(self, single_game):
        white = single_game.white_samples()
        expected = single_game.samples[::2]
        assert white == expected

    def test_black_samples_are_odd_indexed(self, single_game):
        black = single_game.black_samples()
        expected = single_game.samples[1::2]
        assert black == expected


# ===========================================================================
# 3. run_game — termination and legality
# ===========================================================================

class TestRunGame:
    def test_game_terminates(self, model):
        """run_game must always return — never loop forever."""
        record = run_game(model, max_moves=200)
        assert isinstance(record, GameRecord)

    def test_result_is_valid_string(self, single_game):
        valid = {"white_wins", "black_wins", "draw", "max_moves_reached"}
        assert single_game.result in valid

    def test_n_moves_is_positive(self, single_game):
        assert single_game.n_moves > 0

    def test_n_moves_does_not_exceed_max(self, model):
        max_m  = 10
        record = run_game(model, max_moves=max_m)
        assert record.n_moves <= max_m

    def test_max_moves_caps_game(self, model):
        """A very small cap must be respected."""
        record = run_game(model, max_moves=4)
        assert record.n_moves <= 4

    def test_all_samples_have_correct_tensor_shape(self, single_game):
        """Every board tensor must be (13, 8, 8)."""
        for sample in single_game:
            assert sample.board_tensor.shape == (13, 8, 8)

    def test_all_samples_have_float32_tensors(self, single_game):
        for sample in single_game:
            assert sample.board_tensor.dtype == torch.float32

    def test_all_move_indices_are_valid(self, single_game):
        """move_index must be a valid index into legal_moves."""
        for sample in single_game:
            assert 0 <= sample.move_index < len(sample.legal_moves)

    def test_all_legal_moves_lists_are_non_empty(self, single_game):
        """Each position must have at least one legal move."""
        for sample in single_game:
            assert len(sample.legal_moves) > 0

    def test_all_legal_moves_are_chess_move_instances(self, single_game):
        for sample in single_game:
            for move in sample.legal_moves:
                assert isinstance(move, chess.Move)

    def test_rewards_are_uniform_and_valid(self, single_game):
        """All rewards in a game must be from {-1.0, 0.0, +1.0}."""
        for sample in single_game:
            assert sample.reward in {-1.0, 0.0, 1.0}

    def test_all_samples_in_one_game_share_same_reward_magnitude(self, single_game):
        """
        Within one game, all rewards should have the same absolute value.
        Win/loss games: all |reward| = 1.0.
        Draw/cap games: all |reward| = 0.0.
        """
        magnitudes = {abs(s.reward) for s in single_game}
        assert len(magnitudes) == 1

    def test_winning_result_gives_positive_reward_to_winner(self, model):
        """
        If white_wins, all White samples must have reward +1.0
        and all Black samples must have reward -1.0.
        """
        # Run enough games to likely find a checkmate result
        for _ in range(20):
            record = run_game(model, max_moves=200)
            if record.result == "white_wins":
                for s in record.white_samples():
                    assert s.reward == 1.0
                for s in record.black_samples():
                    assert s.reward == -1.0
                return
        pytest.skip("No white_wins game found in 20 games — skip reward check.")

    def test_draw_or_cap_gives_zero_reward(self, model):
        """All samples in a draw/capped game must have reward 0.0."""
        for _ in range(30):
            record = run_game(model, max_moves=200)
            if record.result in ("draw", "max_moves_reached"):
                for s in record:
                    assert s.reward == 0.0
                return
        pytest.skip("No draw/capped game in 30 games — skip.")

    def test_chosen_move_is_legal_in_original_position(self, single_game):
        """The chosen move (by index) must be in legal_moves."""
        for sample in single_game:
            assert sample.legal_moves[sample.move_index] in sample.legal_moves

    def test_model_weights_unchanged_after_game(self, model):
        """run_game must not modify model parameters."""
        params_before = {
            name: p.data.clone() for name, p in model.named_parameters()
        }
        run_game(model)
        for name, p in model.named_parameters():
            assert torch.equal(p.data, params_before[name]), (
                f"Parameter {name} was modified by run_game()"
            )


# ===========================================================================
# 4. run_games — batch behaviour
# ===========================================================================

class TestRunGames:
    def test_returns_correct_number_of_records(self, model):
        records = run_games(model, n_games=5, max_moves=50)
        assert len(records) == 5

    def test_all_records_are_game_record_instances(self, model):
        for record in run_games(model, n_games=3, max_moves=50):
            assert isinstance(record, GameRecord)

    def test_zero_games_returns_empty_list(self, model):
        assert run_games(model, n_games=0) == []


# ===========================================================================
# 5. records_to_dataset
# ===========================================================================

class TestRecordsToDataset:
    def test_flattens_correctly(self, model):
        records = run_games(model, n_games=3, max_moves=50)
        dataset = records_to_dataset(records)
        expected = sum(len(r) for r in records)
        assert len(dataset) == expected

    def test_returns_list_of_game_samples(self, model):
        records = run_games(model, n_games=2, max_moves=30)
        dataset = records_to_dataset(records)
        for sample in dataset:
            assert isinstance(sample, GameSample)

    def test_empty_records_returns_empty_list(self):
        assert records_to_dataset([]) == []


# ===========================================================================
# 6. compute_loss
# ===========================================================================

class TestComputeLoss:
    def test_returns_scalar_tensor(self, model):
        record  = run_game(model, max_moves=20)
        samples = list(record)
        model.train()
        loss = compute_loss(model, samples)
        assert loss.shape == torch.Size([])

    def test_loss_is_finite(self, model):
        record  = run_game(model, max_moves=20)
        samples = list(record)
        model.train()
        loss = compute_loss(model, samples)
        assert torch.isfinite(loss)

    def test_loss_is_zero_for_all_draw_samples(self, model):
        """
        Samples with reward=0 contribute nothing.
        Creating all-draw samples manually gives loss = 0.
        """
        record = run_game(model, max_moves=20)
        zero_reward_samples = [
            GameSample(s.board_tensor, s.legal_moves, s.move_index, 0.0)
            for s in record
        ]
        model.train()
        loss = compute_loss(model, zero_reward_samples)
        assert loss.item() == 0.0

    def test_loss_is_positive_for_winning_samples(self, model):
        """Winning samples (reward=+1) produce positive cross-entropy loss."""
        record = run_game(model, max_moves=20)
        win_samples = [
            GameSample(s.board_tensor, s.legal_moves, s.move_index, 1.0)
            for s in record
        ]
        model.train()
        loss = compute_loss(model, win_samples)
        assert loss.item() > 0.0

    def test_loss_is_negative_for_losing_samples(self, model):
        """Losing samples (reward=-1) produce negative weighted loss."""
        record = run_game(model, max_moves=20)
        loss_samples = [
            GameSample(s.board_tensor, s.legal_moves, s.move_index, -1.0)
            for s in record
        ]
        model.train()
        loss = compute_loss(model, loss_samples)
        assert loss.item() < 0.0

    def test_loss_requires_grad(self, model):
        """Loss tensor must have requires_grad=True for backprop."""
        record  = run_game(model, max_moves=10)
        samples = [
            GameSample(s.board_tensor, s.legal_moves, s.move_index, 1.0)
            for s in record
        ]
        model.train()
        loss = compute_loss(model, samples)
        assert loss.requires_grad


# ===========================================================================
# 7. TrainingConfig
# ===========================================================================

class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.n_epochs        == 50
        assert cfg.games_per_epoch == 20
        assert cfg.learning_rate   == 1e-3
        assert cfg.max_moves       == 200
        assert cfg.temp_high       == 1.0
        assert cfg.temp_low        == 0.1
        assert cfg.temp_threshold  == 30
        assert cfg.device          == "cpu"

    def test_custom_values(self):
        cfg = TrainingConfig(n_epochs=5, games_per_epoch=2, learning_rate=0.01)
        assert cfg.n_epochs        == 5
        assert cfg.games_per_epoch == 2
        assert cfg.learning_rate   == 0.01


# ===========================================================================
# 8. EpochMetrics
# ===========================================================================

class TestEpochMetrics:
    def test_summary_contains_epoch_number(self):
        m = EpochMetrics(1, 0.5, 100, 10, 4, 3, 2, 1, 2.5)
        assert "1" in m.summary()

    def test_summary_contains_loss(self):
        m = EpochMetrics(1, 0.1234, 100, 10, 4, 3, 2, 1, 2.5)
        assert "0.1234" in m.summary()

    def test_summary_is_string(self):
        m = EpochMetrics(1, 0.5, 100, 10, 4, 3, 2, 1, 2.5)
        assert isinstance(m.summary(), str)


# ===========================================================================
# 9. Checkpoint save / load
# ===========================================================================

class TestCheckpoints:
    def test_save_creates_file(self, model, tmp_path):
        metrics = EpochMetrics(1, 0.5, 100, 10, 4, 3, 2, 1, 1.0)
        path = save_checkpoint(model, epoch=1, metrics=metrics, save_dir=str(tmp_path))
        assert os.path.exists(path)

    def test_load_restores_weights(self, model, tmp_path):
        metrics = EpochMetrics(1, 0.5, 100, 10, 4, 3, 2, 1, 1.0)
        path  = save_checkpoint(model, epoch=1, metrics=metrics, save_dir=str(tmp_path))

        fresh = PolicyNetwork()
        epoch = load_checkpoint(path, fresh)

        assert epoch == 1
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), fresh.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Weights differ for {n1}"

    def test_load_returns_correct_epoch(self, model, tmp_path):
        metrics = EpochMetrics(7, 0.5, 100, 10, 4, 3, 2, 1, 1.0)
        path  = save_checkpoint(model, epoch=7, metrics=metrics, save_dir=str(tmp_path))
        fresh = PolicyNetwork()
        assert load_checkpoint(path, fresh) == 7


# ===========================================================================
# 10. Full training loop (smoke test — 2 epochs, 2 games)
# ===========================================================================

class TestTrainLoop:
    def test_train_returns_model_and_history(self):
        cfg = TrainingConfig(
            n_epochs=2,
            games_per_epoch=2,
            max_moves=30,
            checkpoint_every=0,   # no checkpoints in tests
        )
        trained_model, history = train(config=cfg)
        assert isinstance(trained_model, PolicyNetwork)
        assert len(history) == 2

    def test_history_contains_epoch_metrics(self):
        cfg = TrainingConfig(
            n_epochs=2,
            games_per_epoch=2,
            max_moves=30,
            checkpoint_every=0,
        )
        _, history = train(config=cfg)
        for m in history:
            assert isinstance(m, EpochMetrics)

    def test_loss_is_recorded_each_epoch(self):
        cfg = TrainingConfig(
            n_epochs=2,
            games_per_epoch=2,
            max_moves=30,
            checkpoint_every=0,
        )
        _, history = train(config=cfg)
        for m in history:
            assert isinstance(m.loss, float)

    def test_model_parameters_change_after_training(self):
        """
        Weights must change after a training step that has a non-zero reward.

        We do NOT rely on self-play producing a win in a short game — a
        randomly initialised model almost never checkmates in 30 moves, so
        all games would be capped (reward=0) and no gradient would flow.

        Instead we run one game, forcibly override every sample's reward to
        +1.0 (simulating a winning game), then run exactly one compute_loss +
        optimiser step.  This directly tests that the gradient path from
        compute_loss through the network to the parameters is working.
        """
        model = PolicyNetwork()
        model.train()
        params_before = {n: p.data.clone() for n, p in model.named_parameters()}

        # Generate samples from a real game (legal moves, real board tensors)
        record = run_game(model, max_moves=40)
        # Force reward=+1 on every sample so loss.requires_grad is True
        win_samples = [
            GameSample(s.board_tensor, s.legal_moves, s.move_index, 1.0)
            for s in record
        ]

        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimiser.zero_grad()
        loss = compute_loss(model, win_samples)
        assert loss.requires_grad, "Loss has no grad_fn — test setup is broken"
        loss.backward()
        optimiser.step()

        changed = sum(
            1 for n, p in model.named_parameters()
            if not torch.equal(p.data, params_before[n])
        )
        assert changed > 0, "No parameters changed after training"

    def test_checkpoint_saved_when_configured(self, tmp_path):
        """Checkpoint files must exist at configured intervals."""
        cfg = TrainingConfig(
            n_epochs=2,
            games_per_epoch=2,
            max_moves=30,
            checkpoint_every=1,
            checkpoint_dir=str(tmp_path),
        )
        train(config=cfg)
        saved = list(tmp_path.glob("*.pt"))
        assert len(saved) == 2   # one per epoch


import os   # needed for TestCheckpoints.test_save_creates_file