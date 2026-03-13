"""
self_play.py
------------
Generates training data by running the policy network against itself.

Key behaviours
--------------
• En passant — python-chess handles it natively; `board.legal_moves` always
  includes legal en-passant captures, so no special code is needed.

• Draw detection (applied every half-move, before the move cap):
    - Stalemate
    - Insufficient material
    - 50-move rule  (halfmove clock ≥ 100 half-moves)
    - Threefold repetition
  All of these terminate the game immediately with reward 0.

• Move cap  — if the game is still running after `max_moves` half-moves,
  it is sent to Stockfish for evaluation.  Stockfish returns a centipawn
  score from White's perspective.  If |score| >= STOCKFISH_WIN_THRESHOLD
  (default 100 cp = 1.0 pawn) the winning side gets +1.0 and the losing
  side -1.0; otherwise both sides get 0.0 (draw).
  If Stockfish is not installed or evaluation fails, the capped game is
  treated as a draw (safe fallback).

• All game-ending rules that python-chess enforces (checkmate, 75-move rule,
  fivefold repetition) are also caught — they happen naturally via
  `board.is_game_over()`.
"""

from __future__ import annotations

import chess
import chess.engine
import torch
import os
import shutil
from dataclasses import dataclass, field
from typing import NamedTuple

from src.environment.chess_env import ChessEnv
from src.environment.board_encoder import encode_board
from src.models.policy_network import PolicyNetwork


# ---------------------------------------------------------------------------
# Stockfish config
# ---------------------------------------------------------------------------

# Centipawn threshold for declaring a winner at move-cap.
# 100 cp = 1 pawn advantage = "decisive" for our purposes.
STOCKFISH_WIN_THRESHOLD_CP: int = 100

# Time limit (seconds) given to Stockfish per position at move-cap.
STOCKFISH_TIME_LIMIT: float = 0.1

def _find_stockfish() -> str | None:
    """Return the path to the Stockfish binary, or None if not found."""
    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    return shutil.which("stockfish")


# Resolve once at import — avoids filesystem hit on every capped game.
_SF_PATH: str | None = _find_stockfish()
if _SF_PATH:
    print(f"[self_play] Stockfish found: {_SF_PATH}")
else:
    print("[self_play] Stockfish NOT found — capped games scored as draws. "
          "Set STOCKFISH_PATH env variable to your stockfish.exe path.")


def _stockfish_evaluate(fen: str) -> float | None:
    """
    Run Stockfish on the given FEN and return the score from White's POV
    in centipawns, or None if Stockfish is unavailable or fails.

    Returns
    -------
    float | None
        Centipawn score (positive = White winning, negative = Black winning).
        Mate scores are clamped to ±100_000.
        None if Stockfish is not available or evaluation raises an exception.
    """
    sf_path = _SF_PATH
    if sf_path is None:
        return None

    try:
        with chess.engine.SimpleEngine.popen_uci(sf_path) as engine:
            board  = chess.Board(fen)
            result = engine.analyse(
                board,
                chess.engine.Limit(time=STOCKFISH_TIME_LIMIT),
            )
            score = result["score"].white()   # PovScore from White's POV
            if score.is_mate():
                # Mate in N — treat as decisive
                return 100_000.0 if score.mate() > 0 else -100_000.0
            return float(score.score())       # centipawns
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class GameSample(NamedTuple):
    """
    One training sample produced from a single move in a self-play game.

    Fields
    ------
    board_tensor : torch.Tensor  shape (13, 8, 8) — board state BEFORE move.
    legal_moves  : list[chess.Move]
    move_index   : int            index into legal_moves of the chosen move.
    reward       : float          assigned after game ends.
    """
    board_tensor : torch.Tensor
    legal_moves  : list
    move_index   : int
    reward       : float


@dataclass
class GameRecord:
    """
    Complete record of one self-play game.

    result values
    -------------
    'white_wins'        — checkmate
    'black_wins'        — checkmate
    'draw'              — stalemate / repetition / 50-move / insufficient material
    'max_moves_reached' — hit move cap; Stockfish used for reward assignment
    'max_moves_draw'    — hit cap, Stockfish says roughly equal
    """
    samples       : list[GameSample] = field(default_factory=list)
    result        : str              = "in_progress"
    n_moves       : int              = 0
    stockfish_cp  : float | None     = None
    opponent_type : str              = "self"  # "self" | "RandomAgent" | "CheckpointAgent"

    def __len__(self)  -> int: return len(self.samples)
    def __iter__(self):        return iter(self.samples)

    def white_samples(self) -> list[GameSample]: return self.samples[::2]
    def black_samples(self) -> list[GameSample]: return self.samples[1::2]


# ---------------------------------------------------------------------------
# Temperature schedule
# ---------------------------------------------------------------------------

def get_temperature(
    move_number : int,
    temp_high   : float,
    temp_low    : float,
    threshold   : int,
) -> float:
    """Return sampling temperature based on move number."""
    return temp_high if move_number < threshold else temp_low


# ---------------------------------------------------------------------------
# Early-draw detection
# ---------------------------------------------------------------------------

def _is_early_draw(board: chess.Board) -> bool:
    """
    Return True if the position is a draw by any rule that python-chess can
    detect *before* the move cap.

    Covers:
      • Stalemate
      • Insufficient material
      • 50-move rule   (halfmove_clock >= 100 means 50 full moves without
                        pawn move or capture — python-chess uses half-moves)
      • Threefold repetition
    """
    return (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_fifty_moves()          # halfmove clock ≥ 100
        or board.is_repetition(count=3)    # threefold repetition
    )


# ---------------------------------------------------------------------------
# Core: single game
# ---------------------------------------------------------------------------

def run_game(
    model          : PolicyNetwork,
    opponent       = None,
    model_color    : bool  = chess.WHITE,
    max_moves      : int   = 200,
    temp_high      : float = 1.0,
    temp_low       : float = 0.1,
    temp_threshold : int   = 30,
    device         : str   = "cpu",
    use_stockfish  : bool  = True,
) -> GameRecord:
    """
    Play one game and return a GameRecord of training samples.

    Opponent pool support
    ---------------------
    Pass any agent with a ``select_move(board) -> chess.Move`` interface as
    ``opponent``.  When an opponent is supplied:

      • The training model plays as ``model_color`` (default White).
      • The opponent plays the other colour.
      • Only the model's moves are stored as GameSamples — the opponent's
        moves are applied silently and never trained on.
      • Rewards are assigned from the model's perspective:
          model wins  →  all model samples get +1.0
          model loses →  all model samples get -1.0
          draw        →  all model samples get  0.0

    When ``opponent=None`` (default), the game is pure self-play: the model
    plays both colours and samples from both sides are stored with per-side
    rewards (same behaviour as before).

    En passant
    ----------
    Fully supported — python-chess includes legal en-passant captures in
    ``board.legal_moves`` automatically, so no special handling is needed.

    Move cap + Stockfish
    --------------------
    If the game reaches ``max_moves`` half-moves without a decisive result,
    Stockfish evaluates the final position.  If the score exceeds
    STOCKFISH_WIN_THRESHOLD_CP centipawns the winning side is awarded ±1.0.
    Otherwise both sides receive 0.0.  Falls back to draw if Stockfish is
    unavailable.

    Parameters
    ----------
    model          : PolicyNetwork
    opponent       : agent | None
        Any object with ``select_move(board) -> chess.Move``, or None for
        pure self-play.  Use OpponentPool.sample() to get this value.
    model_color    : bool
        chess.WHITE or chess.BLACK.  Which colour the training model plays
        when an opponent is present.  Ignored in self-play (opponent=None).
    max_moves      : int    Half-move cap (default 200).
    temp_high      : float  Temperature for first ``temp_threshold`` moves.
    temp_low       : float  Temperature after ``temp_threshold`` moves.
    temp_threshold : int    Half-move number to switch temperature.
    device         : str    Torch device.
    use_stockfish  : bool   If False, skip Stockfish and treat cap as draw.

    Returns
    -------
    GameRecord
    """
    is_self_play = (opponent is None)

    model.eval()
    env    = ChessEnv()
    record = GameRecord()
    env.reset()

    # raw_samples: (board_tensor, legal_moves, move_index, white_to_move)
    # Only the training model's moves are stored here.
    raw_samples: list[tuple[torch.Tensor, list, int, bool]] = []

    for move_number in range(max_moves):

        # ── Check for natural game end ──────────────────────────────────
        if env.is_game_over():
            break

        # ── Check for early draw (repetition / 50-move / stalemate / material) ──
        if _is_early_draw(env.board):
            break

        legal = env.get_legal_moves()
        if not legal:
            break

        white_to_move   = (env.board.turn == chess.WHITE)
        is_model_turn   = is_self_play or (env.board.turn == model_color)

        if is_model_turn:
            # ── Model's turn: encode, score, sample, record ─────────────
            state_np = encode_board(env.board)                         # (13,8,8)
            state_t  = torch.tensor(state_np, dtype=torch.float32, device=device)
            input_t  = state_t.unsqueeze(0)                            # (1,13,8,8)

            temp = get_temperature(move_number, temp_high, temp_low, temp_threshold)

            with torch.no_grad():
                chosen_move = model.select_move(input_t, legal, temperature=temp)

            move_index = legal.index(chosen_move)
            raw_samples.append((state_t, legal, move_index, white_to_move))

        else:
            # ── Opponent's turn: just pick a move, no sample stored ──────
            try:
                chosen_move = opponent.select_move(env.board)
            except Exception:
                # Opponent failure (e.g. corrupt checkpoint) — fall back to
                # a random legal move so the game can still complete.
                import random as _random
                chosen_move = _random.choice(legal)

        # ── Apply move (en passant handled automatically by python-chess) ──
        env.step(chosen_move)

    # -----------------------------------------------------------------------
    # Determine outcome and assign rewards
    # -----------------------------------------------------------------------

    if env.is_game_over():
        outcome      = env.get_game_result()   # 'white_wins' | 'black_wins' | 'draw'
        result_label = outcome
        stockfish_cp = None

    elif _is_early_draw(env.board):
        outcome      = "draw"
        result_label = "draw"
        stockfish_cp = None

    else:
        # ── Move cap reached ── ask Stockfish ──────────────────────────
        result_label = "max_moves_reached"
        stockfish_cp = None

        if use_stockfish:
            stockfish_cp = _stockfish_evaluate(env.board.fen())

        if stockfish_cp is not None:
            if stockfish_cp >= STOCKFISH_WIN_THRESHOLD_CP:
                outcome = "white_wins"
            elif stockfish_cp <= -STOCKFISH_WIN_THRESHOLD_CP:
                outcome = "black_wins"
            else:
                outcome = "draw"
                result_label = "max_moves_draw"
        else:
            outcome      = "draw"
            result_label = "max_moves_draw"

    # -----------------------------------------------------------------------
    # Map outcome → per-sample reward
    # -----------------------------------------------------------------------

    if is_self_play:
        # Both sides' moves are stored — assign per-side rewards as before
        if outcome == "white_wins":
            white_reward, black_reward =  1.0, -1.0
        elif outcome == "black_wins":
            white_reward, black_reward = -1.0,  1.0
        else:
            white_reward, black_reward =  0.0,  0.0

        for (board_tensor, legal_moves, move_index, white_to_move) in raw_samples:
            reward = white_reward if white_to_move else black_reward
            record.samples.append(GameSample(
                board_tensor=board_tensor,
                legal_moves=legal_moves,
                move_index=move_index,
                reward=reward,
            ))

    else:
        # Only the model's moves are stored — assign one reward from its POV
        model_won = (
            (outcome == "white_wins" and model_color == chess.WHITE) or
            (outcome == "black_wins" and model_color == chess.BLACK)
        )
        model_lost = (
            (outcome == "black_wins" and model_color == chess.WHITE) or
            (outcome == "white_wins" and model_color == chess.BLACK)
        )

        if model_won:
            model_reward = 1.0
        elif model_lost:
            model_reward = -1.0
        else:
            model_reward = 0.0

        for (board_tensor, legal_moves, move_index, _white_to_move) in raw_samples:
            record.samples.append(GameSample(
                board_tensor=board_tensor,
                legal_moves=legal_moves,
                move_index=move_index,
                reward=model_reward,
            ))

    record.result        = result_label
    record.n_moves       = len(raw_samples)
    record.stockfish_cp  = stockfish_cp
    record.opponent_type = "self"   # overwritten by run_games when pool is used
    return record


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_games(
    model          : PolicyNetwork,
    n_games        : int   = 10,
    max_moves      : int   = 200,
    temp_high      : float = 1.0,
    temp_low       : float = 0.1,
    temp_threshold : int   = 30,
    device         : str   = "cpu",
    use_stockfish  : bool  = True,
    verbose        : bool  = False,
    opponent_pool  = None,   # OpponentPool | None
) -> list[GameRecord]:
    """
    Run multiple games and return all GameRecords.

    When opponent_pool is provided each game samples a fresh opponent and
    the model alternates White/Black to remove colour bias.
    When opponent_pool is None all games are pure self-play.

    Always prints a per-agent breakdown after all games complete,
    regardless of the verbose flag.
    """
    from collections import defaultdict

    records: list[GameRecord] = []
    # {opponent_type: [wins, losses, draws, capped]}
    agent_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])

    for i in range(n_games):
        if opponent_pool is not None:
            opponent    = opponent_pool.sample()
            model_color = chess.WHITE if i % 2 == 0 else chess.BLACK
        else:
            opponent    = None
            model_color = chess.WHITE   # unused in self-play

        record = run_game(
            model,
            opponent       = opponent,
            model_color    = model_color,
            max_moves      = max_moves,
            temp_high      = temp_high,
            temp_low       = temp_low,
            temp_threshold = temp_threshold,
            device         = device,
            use_stockfish  = use_stockfish,
        )

        # Tag the record with the opponent name — THIS is what was missing
        opp_name             = "self" if opponent is None else type(opponent).__name__
        record.opponent_type = opp_name

        # Tally per-agent outcome: [wins, losses, draws, capped]
        if record.result == "white_wins":
            agent_counts[opp_name][0] += 1
        elif record.result == "black_wins":
            agent_counts[opp_name][1] += 1
        elif record.result in ("draw", "max_moves_draw"):
            agent_counts[opp_name][2] += 1
        else:
            agent_counts[opp_name][3] += 1   # max_moves_reached

        records.append(record)

        if verbose:
            sf_str    = (f"  sf={record.stockfish_cp:+.0f}cp"
                         if record.stockfish_cp is not None else "")
            color_str = "W" if model_color == chess.WHITE else "B"
            print(
                f"  Game {i+1:>3}/{n_games} | "
                f"Opp: {opp_name:<16} | "
                f"Model: {color_str} | "
                f"Moves: {record.n_moves:>3} | "
                f"Result: {record.result}{sf_str}"
            )

    # Always print per-agent summary
    lines = [f"  Games this epoch: {len(records)}"]
    for opp_name, (w, l, d, cap) in sorted(agent_counts.items()):
        n = w + l + d + cap
        lines.append(
            f"    vs {opp_name:<18} {n:>3} games  |  "
            f"W:{w:<3} L:{l:<3} D:{d:<3} Cap:{cap}"
        )
    print("\n".join(lines))

    return records


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------

def records_to_dataset(records: list[GameRecord]) -> list[GameSample]:
    """Flatten a list of GameRecords into a single list of GameSamples."""
    return [sample for record in records for sample in record]