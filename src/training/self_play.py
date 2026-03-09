"""
self_play.py
------------
Generates training data by running the policy network against itself.

Each call to run_game() plays one complete game and returns a GameRecord
containing every (board_tensor, legal_moves, chosen_move_index) triple,
plus the final game result used to assign uniform rewards.

Design decisions
----------------
- Same model plays both sides — the network learns from both White and
  Black perspectives simultaneously, which is the standard AlphaZero
  approach and avoids maintaining two separate model copies at this stage.

- Temperature scheduling — high temperature (1.0) early in the game
  encourages exploration; lower temperature (0.1) in the endgame (after
  move threshold) focuses on exploitation once the position is clearer.

- Uniform reward — every move in a winning game gets +1, every move in
  a losing game gets -1, every move in a draw gets 0.  Simple and stable
  for early training.

- max_moves cap — prevents games from running indefinitely during early
  training when the model plays randomly.  Capped games are scored as draws.

Usage
-----
    from src.models.policy_network import PolicyNetwork
    from src.training.self_play import run_game, run_games

    model = PolicyNetwork()

    # Single game
    record = run_game(model)
    print(f"Moves: {len(record)}, Result: {record.result}")

    # Batch of games
    records = run_games(model, n_games=10)
    dataset = [sample for record in records for sample in record]
"""

from __future__ import annotations

import chess
import torch
from dataclasses import dataclass, field
from typing import NamedTuple

from src.environment.chess_env import ChessEnv
from src.environment.board_encoder import encode_board
from src.models.policy_network import PolicyNetwork


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class GameSample(NamedTuple):
    """
    One training sample produced from a single move in a self-play game.

    Fields
    ------
    board_tensor : torch.Tensor
        Shape (13, 8, 8) — board state BEFORE the move was played.
    legal_moves : list[chess.Move]
        All legal moves available in this position.
    move_index : int
        Index into legal_moves of the move that was chosen.
    reward : float
        Uniform game reward assigned after the game ends:
          +1.0 for the side that won
          -1.0 for the side that lost
           0.0 for a draw or capped game
    """
    board_tensor : torch.Tensor
    legal_moves  : list
    move_index   : int
    reward       : float


@dataclass
class GameRecord:
    """
    Complete record of one self-play game.

    Attributes
    ----------
    samples : list[GameSample]
        Ordered list of all samples produced during this game.
    result : str
        One of: 'white_wins', 'black_wins', 'draw', 'max_moves_reached'.
    n_moves : int
        Total number of half-moves (plies) played.
    """
    samples  : list[GameSample] = field(default_factory=list)
    result   : str              = "in_progress"
    n_moves  : int              = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def white_samples(self) -> list[GameSample]:
        """Return samples from White's moves only (even indices: 0, 2, 4…)."""
        return self.samples[::2]

    def black_samples(self) -> list[GameSample]:
        """Return samples from Black's moves only (odd indices: 1, 3, 5…)."""
        return self.samples[1::2]


# ---------------------------------------------------------------------------
# Temperature schedule
# ---------------------------------------------------------------------------

def get_temperature(move_number: int, temp_high: float, temp_low: float, threshold: int) -> float:
    """
    Return the sampling temperature for a given move number.

    Before `threshold` moves: return temp_high (exploration).
    At or after `threshold`  : return temp_low  (exploitation).

    Parameters
    ----------
    move_number : int   Current half-move number (0-indexed).
    temp_high   : float Temperature for early game (default 1.0).
    temp_low    : float Temperature for late game  (default 0.1).
    threshold   : int   Move number at which to switch to temp_low.

    Returns
    -------
    float
    """
    return temp_high if move_number < threshold else temp_low


# ---------------------------------------------------------------------------
# Core: single game
# ---------------------------------------------------------------------------

def run_game(
    model       : PolicyNetwork,
    max_moves   : int   = 200,
    temp_high   : float = 1.0,
    temp_low    : float = 0.1,
    temp_threshold: int = 30,
    device      : str   = "cpu",
) -> GameRecord:
    """
    Play one complete self-play game and return a GameRecord.

    The same model plays both sides.  Moves are sampled from the softmax
    policy distribution with temperature scheduling.

    Parameters
    ----------
    model : PolicyNetwork
        The policy network — used in eval mode, weights are not modified.
    max_moves : int
        Maximum number of half-moves before the game is declared a draw.
        Prevents infinite games during early training (default: 200).
    temp_high : float
        Sampling temperature before temp_threshold moves (default: 1.0).
    temp_low : float
        Sampling temperature at and after temp_threshold moves (default: 0.1).
    temp_threshold : int
        Half-move number at which temperature drops to temp_low (default: 30).
    device : str
        Torch device to run inference on (default: 'cpu').

    Returns
    -------
    GameRecord
        Contains all GameSamples and the final result string.
    """
    model.eval()
    env    = ChessEnv()
    record = GameRecord()
    env.reset()

    # Collect raw samples (without reward — assigned after game ends)
    raw_samples: list[tuple[torch.Tensor, list, int, bool]] = []
    # Each entry: (board_tensor, legal_moves, move_index, white_to_move)

    for move_number in range(max_moves):

        if env.is_game_over():
            break

        legal = env.get_legal_moves()
        if not legal:
            break

        # Encode current board state
        state_np  = encode_board(env.board)                            # (13, 8, 8)
        state_t   = torch.tensor(state_np, dtype=torch.float32, device=device)
        input_t   = state_t.unsqueeze(0)                               # (1, 13, 8, 8)

        # Determine temperature for this move
        temp = get_temperature(move_number, temp_high, temp_low, temp_threshold)

        # Sample a move from the policy
        with torch.no_grad():
            chosen_move = model.select_move(input_t, legal, temperature=temp)

        # Record whose turn it is BEFORE applying the move
        white_to_move = (env.board.turn == chess.WHITE)

        # Find the index of the chosen move in legal_moves
        move_index = legal.index(chosen_move)

        # Store raw sample (reward assigned later)
        raw_samples.append((state_t, legal, move_index, white_to_move))

        # Apply move to environment
        env.step(chosen_move)

    # ------------------------------------------------------------------
    # Determine game result and assign uniform rewards
    # ------------------------------------------------------------------
    result = env.get_game_result()

    if env.is_game_over():
        outcome = env.get_game_result()
    else:
        outcome = "draw"   # max_moves reached → treat as draw

    result_label = outcome if env.is_game_over() else "max_moves_reached"

    # Map result to per-side reward
    # white_reward is the reward for the White side;
    # Black gets the opposite.
    if outcome == "white_wins":
        white_reward =  1.0
        black_reward = -1.0
    elif outcome == "black_wins":
        white_reward = -1.0
        black_reward =  1.0
    else:
        white_reward =  0.0
        black_reward =  0.0

    # Assemble final GameSamples with rewards
    for (board_tensor, legal_moves, move_index, white_to_move) in raw_samples:
        reward = white_reward if white_to_move else black_reward
        record.samples.append(
            GameSample(
                board_tensor=board_tensor,
                legal_moves=legal_moves,
                move_index=move_index,
                reward=reward,
            )
        )

    record.result  = result_label
    record.n_moves = len(raw_samples)

    return record


# ---------------------------------------------------------------------------
# Convenience: batch of games
# ---------------------------------------------------------------------------

def run_games(
    model     : PolicyNetwork,
    n_games   : int   = 10,
    max_moves : int   = 200,
    temp_high : float = 1.0,
    temp_low  : float = 0.1,
    temp_threshold: int = 30,
    device    : str   = "cpu",
    verbose   : bool  = False,
) -> list[GameRecord]:
    """
    Run multiple self-play games and return all GameRecords.

    Parameters
    ----------
    model     : PolicyNetwork
    n_games   : int   Number of games to play (default: 10).
    max_moves : int   Per-game move cap (default: 200).
    temp_high : float Early-game temperature (default: 1.0).
    temp_low  : float Late-game temperature  (default: 0.1).
    temp_threshold : int  Move number to switch temperature (default: 30).
    device    : str   Torch device (default: 'cpu').
    verbose   : bool  Print a summary line per game if True.

    Returns
    -------
    list[GameRecord]
    """
    records = []
    for i in range(n_games):
        record = run_game(
            model,
            max_moves=max_moves,
            temp_high=temp_high,
            temp_low=temp_low,
            temp_threshold=temp_threshold,
            device=device,
        )
        records.append(record)
        if verbose:
            print(
                f"  Game {i + 1:>3}/{n_games} | "
                f"Moves: {record.n_moves:>3} | "
                f"Result: {record.result}"
            )
    return records


# ---------------------------------------------------------------------------
# Convenience: flatten records into a flat sample list
# ---------------------------------------------------------------------------

def records_to_dataset(records: list[GameRecord]) -> list[GameSample]:
    """
    Flatten a list of GameRecords into a single list of GameSamples.

    Useful for feeding directly into a training loop.

    Parameters
    ----------
    records : list[GameRecord]

    Returns
    -------
    list[GameSample]
    """
    return [sample for record in records for sample in record]