"""
test_policy_network.py
----------------------
Unit tests for PolicyNetwork and encode_move()
(src/models/policy_network.py).

Run with:
    pytest tests/test_policy_network.py -v
"""

import chess
import pytest
import torch
import torch.nn as nn

from src.models.policy_network import PolicyNetwork, encode_move
from src.environment.board_encoder import encode_board


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def board_tensor(fen=None) -> torch.Tensor:
    """Return a (1, 13, 8, 8) float32 tensor for the given FEN (or start pos)."""
    board = chess.Board(fen) if fen else chess.Board()
    return torch.tensor(encode_board(board)).unsqueeze(0)


def legal_moves(fen=None) -> list:
    """Return list of legal moves for the given FEN (or start pos)."""
    board = chess.Board(fen) if fen else chess.Board()
    return list(board.legal_moves)


@pytest.fixture
def net():
    """Fresh PolicyNetwork in eval mode before each test."""
    model = PolicyNetwork()
    model.eval()
    return model


@pytest.fixture
def start_tensor():
    return board_tensor()


@pytest.fixture
def start_moves():
    return legal_moves()


# ===========================================================================
# 1. encode_move
# ===========================================================================

class TestEncodeMove:
    def test_output_shape(self):
        """encode_move must return a 130-dim vector."""
        assert encode_move(chess.Move.from_uci("e2e4")).shape == (130,)

    def test_output_dtype(self):
        """encode_move must return float32."""
        assert encode_move(chess.Move.from_uci("e2e4")).dtype == torch.float32

    def test_only_zeros_and_ones_for_quiet_move(self):
        """A quiet (non-promotion) move contains only 0.0 and 1.0."""
        vec = encode_move(chess.Move.from_uci("e2e4"))
        assert set(vec.unique().tolist()).issubset({0.0, 1.0})

    def test_exactly_two_ones_for_quiet_move(self):
        """A quiet move sets exactly 2 bits: source + destination."""
        assert encode_move(chess.Move.from_uci("e2e4")).sum().item() == 2.0

    def test_source_square_encoded_correctly(self):
        """
        e2 = rank 1, file 4 → index 1*8+4 = 12.
        Source one-hot lives in indices 0-63.
        """
        assert encode_move(chess.Move.from_uci("e2e4"))[12].item() == 1.0

    def test_destination_square_encoded_correctly(self):
        """
        e4 = rank 3, file 4 → index 3*8+4 = 28 → offset 64 = index 92.
        Destination one-hot lives in indices 64-127.
        """
        assert encode_move(chess.Move.from_uci("e2e4"))[64 + 28].item() == 1.0

    def test_promotion_flag_zero_for_quiet_move(self):
        """Index 128 must be 0.0 for a non-promotion move."""
        assert encode_move(chess.Move.from_uci("e2e4"))[128].item() == 0.0

    def test_promotion_flag_set_for_promotion(self):
        """Index 128 must be 1.0 for a promotion move."""
        assert encode_move(chess.Move.from_uci("e7e8q"))[128].item() == 1.0

    def test_promotion_piece_normalised_queen(self):
        """Queen promotion (piece type 5) normalises to 1.0."""
        assert pytest.approx(encode_move(chess.Move.from_uci("e7e8q"))[129].item()) == 1.0

    def test_promotion_piece_normalised_knight(self):
        """Knight promotion (piece type 2) normalises to 0.4."""
        assert pytest.approx(encode_move(chess.Move.from_uci("e7e8n"))[129].item()) == 2 / 5

    def test_different_moves_produce_different_vectors(self):
        """Two distinct moves must not produce identical vectors."""
        v1 = encode_move(chess.Move.from_uci("e2e4"))
        v2 = encode_move(chess.Move.from_uci("d2d4"))
        assert not torch.equal(v1, v2)

    def test_same_move_is_deterministic(self):
        """Same move always produces the same vector."""
        m = chess.Move.from_uci("g1f3")
        assert torch.equal(encode_move(m), encode_move(m))


# ===========================================================================
# 2. Network construction
# ===========================================================================

class TestNetworkConstruction:
    def test_instantiates_without_error(self):
        PolicyNetwork()

    def test_default_embedding_dim(self, net):
        assert net.embedding_dim == 128

    def test_default_move_feature_dim(self, net):
        assert net.move_feature_dim == 130

    def test_has_three_conv_layers(self, net):
        assert isinstance(net.conv1, nn.Conv2d)
        assert isinstance(net.conv2, nn.Conv2d)
        assert isinstance(net.conv3, nn.Conv2d)

    def test_conv_channel_progression(self, net):
        """13 → 32 → 64 → 64 as specified."""
        assert net.conv1.in_channels  == 13 and net.conv1.out_channels == 32
        assert net.conv2.in_channels  == 32 and net.conv2.out_channels == 64
        assert net.conv3.in_channels  == 64 and net.conv3.out_channels == 64

    def test_conv_kernel_size(self, net):
        for layer in (net.conv1, net.conv2, net.conv3):
            assert layer.kernel_size == (3, 3)

    def test_conv_padding_preserves_spatial_dims(self, net):
        for layer in (net.conv1, net.conv2, net.conv3):
            assert layer.padding == (1, 1)

    def test_fc1_dimensions(self, net):
        assert net.fc1.in_features == 64 * 8 * 8 and net.fc1.out_features == 256

    def test_fc2_dimensions(self, net):
        assert net.fc2.in_features == 256 and net.fc2.out_features == 128

    def test_move_projector_exists(self, net):
        """Dot-product scoring requires a move_projector Linear layer."""
        assert isinstance(net.move_projector, nn.Linear)

    def test_move_projector_input_size(self, net):
        """move_projector must accept 130-dim move feature vectors."""
        assert net.move_projector.in_features == 130

    def test_move_projector_output_matches_embedding_dim(self, net):
        """move_projector output must equal embedding_dim (128) for dot product."""
        assert net.move_projector.out_features == net.embedding_dim

    def test_move_projector_has_no_bias(self, net):
        """
        bias=False on move_projector — a bias would shift all move scores
        equally and cancel out of the softmax, wasting parameters.
        """
        assert net.move_projector.bias is None

    def test_dot_product_scoring_is_correct(self, net):
        """
        Manually verify the dot-product:
          logit = move_proj @ board_emb.T
        Using a hand-crafted embedding and projection to confirm the
        forward pass computes exactly this operation.
        """
        board = chess.Board()
        t = board_tensor()
        moves = list(board.legal_moves)

        with torch.no_grad():
            # Get board embedding
            emb = net.encode_board(t)                             # (1, 128)

            # Project all moves
            move_vecs = torch.stack([encode_move(m) for m in moves])  # (N, 130)
            move_proj = net.move_projector(move_vecs)                  # (N, 128)

            # Manual dot product
            expected = (move_proj @ emb.T).squeeze(-1).unsqueeze(0)   # (1, N)

            # Network forward pass
            actual = net(t, moves)

        assert torch.allclose(actual, expected, atol=1e-5), (
            "forward() does not match manual dot-product computation"
        )

    def test_trainable_parameter_count_is_positive(self, net):
        assert net.count_parameters() > 0

    def test_repr_contains_key_fields(self, net):
        r = repr(net)
        assert "128" in r and "130" in r and "trainable_params" in r

    def test_custom_embedding_dim_is_respected(self):
        net = PolicyNetwork(embedding_dim=64)
        assert net.embedding_dim == 64
        assert net.fc2.out_features == 64


# ===========================================================================
# 3. Board encoder
# ===========================================================================

class TestBoardEncoder:
    def test_output_shape(self, net, start_tensor):
        assert net.encode_board(start_tensor).shape == (1, 128)

    def test_output_dtype(self, net, start_tensor):
        assert net.encode_board(start_tensor).dtype == torch.float32

    def test_different_positions_give_different_embeddings(self, net):
        t1 = board_tensor()
        t2 = board_tensor("r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2")
        assert not torch.equal(net.encode_board(t1), net.encode_board(t2))

    def test_same_position_is_deterministic(self, net, start_tensor):
        assert torch.equal(
            net.encode_board(start_tensor),
            net.encode_board(start_tensor),
        )

    def test_batch_size_greater_than_one(self, net):
        batch = board_tensor().expand(4, -1, -1, -1)
        assert net.encode_board(batch).shape == (4, 128)


# ===========================================================================
# 4. forward()
# ===========================================================================

class TestForward:
    def test_output_shape_starting_position(self, net, start_tensor, start_moves):
        """20 legal opening moves → output (1, 20)."""
        assert net(start_tensor, start_moves).shape == (1, 20)

    def test_output_shape_matches_num_legal_moves(self, net):
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        t = board_tensor(fen)
        m = legal_moves(fen)
        assert net(t, m).shape == (1, len(m))

    def test_output_dtype(self, net, start_tensor, start_moves):
        assert net(start_tensor, start_moves).dtype == torch.float32

    def test_all_logits_are_finite(self, net, start_tensor, start_moves):
        """No NaN or Inf in any logit."""
        assert torch.isfinite(net(start_tensor, start_moves)).all()

    def test_empty_legal_moves_raises_value_error(self, net, start_tensor):
        with pytest.raises(ValueError, match="empty"):
            net(start_tensor, [])

    def test_logits_not_all_equal(self, net, start_tensor, start_moves):
        """After random init, move scores must vary (std > 0)."""
        logits = net(start_tensor, start_moves).squeeze()
        assert logits.std().item() > 0.0

    def test_softmax_over_logits_sums_to_one(self, net, start_tensor, start_moves):
        logits = net(start_tensor, start_moves)
        probs  = torch.softmax(logits, dim=1)
        assert pytest.approx(probs.sum().item(), abs=1e-5) == 1.0

    def test_no_gradient_required_under_no_grad(self, net, start_tensor, start_moves):
        with torch.no_grad():
            logits = net(start_tensor, start_moves)
        assert not logits.requires_grad

    def test_forward_does_not_modify_legal_moves_list(self, net, start_tensor, start_moves):
        """forward() must not mutate the list it receives."""
        original = list(start_moves)
        net(start_tensor, start_moves)
        assert start_moves == original


# ===========================================================================
# 5. select_move and greedy_move
# ===========================================================================

class TestMoveSampling:
    def test_select_move_returns_a_legal_move(self, net, start_tensor, start_moves):
        assert net.select_move(start_tensor, start_moves) in start_moves

    def test_greedy_move_returns_a_legal_move(self, net, start_tensor, start_moves):
        assert net.greedy_move(start_tensor, start_moves) in start_moves

    def test_greedy_move_is_deterministic(self, net, start_tensor, start_moves):
        assert (
            net.greedy_move(start_tensor, start_moves)
            == net.greedy_move(start_tensor, start_moves)
        )

    def test_select_move_near_zero_temperature_matches_greedy(self, net, start_tensor, start_moves):
        """Temperature ≈ 0 should collapse to greedy selection."""
        greedy  = net.greedy_move(start_tensor, start_moves)
        sampled = net.select_move(start_tensor, start_moves, temperature=1e-9)
        assert sampled == greedy

    def test_select_move_does_not_mutate_board(self):
        """select_move must never push a move onto the chess.Board."""
        net   = PolicyNetwork()
        board = chess.Board()
        fen_before = board.fen()
        t = torch.tensor(encode_board(board)).unsqueeze(0)
        m = list(board.legal_moves)
        net.select_move(t, m)
        assert board.fen() == fen_before

    def test_greedy_picks_highest_logit(self, net, start_tensor, start_moves):
        """greedy_move index must equal argmax of logit tensor."""
        logits      = net(start_tensor, start_moves)
        expected    = start_moves[logits.argmax(dim=1).item()]
        assert net.greedy_move(start_tensor, start_moves) == expected


# ===========================================================================
# 6. Gradient flow and trainability
# ===========================================================================

class TestGradientFlow:
    def test_gradients_reach_all_named_parameters(self, start_tensor, start_moves):
        """
        One backward pass must produce non-None, non-zero gradients for
        every parameter: conv layers, fc layers, and move scorer.
        """
        net = PolicyNetwork()
        net.train()

        logits = net(start_tensor, start_moves)
        target = torch.zeros(1, dtype=torch.long)
        loss   = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()

        for name, param in net.named_parameters():
            assert param.grad is not None,               f"No gradient: {name}"
            assert param.grad.abs().sum().item() > 0.0,  f"Zero gradient: {name}"

    def test_loss_decreases_after_one_sgd_step(self, start_tensor, start_moves):
        """
        A single weight update must reduce cross-entropy loss, confirming
        the network can be trained end-to-end.
        """
        net       = PolicyNetwork()
        net.train()
        optimiser = torch.optim.SGD(net.parameters(), lr=0.01)
        target    = torch.zeros(1, dtype=torch.long)

        logits_before = net(start_tensor, start_moves)
        loss_before   = torch.nn.functional.cross_entropy(logits_before, target)

        optimiser.zero_grad()
        loss_before.backward()
        optimiser.step()

        with torch.no_grad():
            loss_after = torch.nn.functional.cross_entropy(
                net(start_tensor, start_moves), target
            )

        assert loss_after.item() < loss_before.item()

    def test_network_parameters_change_after_update(self, start_tensor, start_moves):
        """
        Parameters must actually change after a gradient step — confirms
        the optimiser is connected and gradients are non-zero.
        """
        net       = PolicyNetwork()
        net.train()
        optimiser = torch.optim.SGD(net.parameters(), lr=0.1)

        params_before = {
            name: param.data.clone()
            for name, param in net.named_parameters()
        }

        logits = net(start_tensor, start_moves)
        loss   = torch.nn.functional.cross_entropy(logits, torch.zeros(1, dtype=torch.long))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        changed = 0
        for name, param in net.named_parameters():
            if not torch.equal(param.data, params_before[name]):
                changed += 1

        assert changed > 0, "No parameters changed after optimiser step"