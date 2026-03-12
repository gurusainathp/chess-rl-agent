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
    # Check explicit environment variable first
    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    # Fall back to PATH lookup
    return shutil.which("stockfish")


# Log once at import time so the very first line of training output confirms
# whether Stockfish is available.  No-op if it isn't — capped games simply
# fall back to draws (safe path in _stockfish_evaluate).
_SF_PATH: str | None = _find_stockfish()
if _SF_PATH:
    print(f"[self_play] Stockfish found: {_SF_PATH}")
else:
    print("[self_play] Stockfish not found — capped games will be scored as draws. "
          "Install via 'sudo apt install stockfish' (Linux) or 'brew install stockfish' (macOS), "
          "or set the STOCKFISH_PATH environment variable.")


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
    samples  : list[GameSample] = field(default_factory=list)
    result   : str              = "in_progress"
    n_moves  : int              = 0
    stockfish_cp: float | None  = None   # set when Stockfish was consulted

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
    max_moves      : int   = 200,
    temp_high      : float = 1.0,
    temp_low       : float = 0.1,
    temp_threshold : int   = 30,
    device         : str   = "cpu",
    use_stockfish  : bool  = True,
) -> GameRecord:
    """
    Play one complete self-play game and return a GameRecord.

    En passant
    ----------
    Fully supported — python-chess includes legal en-passant captures in
    `board.legal_moves` automatically, so no special handling is needed.

    Move cap + Stockfish
    --------------------
    If the game reaches `max_moves` half-moves without a decisive result,
    Stockfish evaluates the final position.  If the score exceeds
    STOCKFISH_WIN_THRESHOLD_CP centipawns the winning side is awarded +1.0
    and the losing side -1.0.  Otherwise both sides receive 0.0 (draw).
    If Stockfish is unavailable the capped game is treated as 0.0 / draw.

    Parameters
    ----------
    model          : PolicyNetwork
    max_moves      : int    Half-move cap (default 200).
    temp_high      : float  Temperature for first `temp_threshold` moves.
    temp_low       : float  Temperature after `temp_threshold` moves.
    temp_threshold : int    Half-move number to switch temperature.
    device         : str    Torch device.
    use_stockfish  : bool   If False, skip Stockfish and treat cap as draw.

    Returns
    -------
    GameRecord
    """
    model.eval()
    env    = ChessEnv()
    record = GameRecord()
    env.reset()

    # raw_samples: list of (board_tensor, legal_moves, move_index, white_to_move)
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

        # ── Encode state ────────────────────────────────────────────────
        state_np = encode_board(env.board)                         # (13,8,8)
        state_t  = torch.tensor(state_np, dtype=torch.float32, device=device)
        input_t  = state_t.unsqueeze(0)                            # (1,13,8,8)

        # ── Temperature ─────────────────────────────────────────────────
        temp = get_temperature(move_number, temp_high, temp_low, temp_threshold)

        # ── Sample move ──────────────────────────────────────────────────
        with torch.no_grad():
            chosen_move = model.select_move(input_t, legal, temperature=temp)

        white_to_move = (env.board.turn == chess.WHITE)
        move_index    = legal.index(chosen_move)
        raw_samples.append((state_t, legal, move_index, white_to_move))

        # ── Apply move (en passant handled automatically by python-chess) ──
        env.step(chosen_move)

    # -----------------------------------------------------------------------
    # Determine outcome and assign rewards
    # -----------------------------------------------------------------------

    if env.is_game_over():
        # Natural end: checkmate or draw by 75-move / fivefold / insufficient
        outcome      = env.get_game_result()   # 'white_wins' | 'black_wins' | 'draw'
        result_label = outcome
        stockfish_cp = None

    elif _is_early_draw(env.board):
        # Draw detected by early-draw check (repetition, 50-move, stalemate)
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
                outcome = "white_wins"        # White is decisively ahead
            elif stockfish_cp <= -STOCKFISH_WIN_THRESHOLD_CP:
                outcome = "black_wins"        # Black is decisively ahead
            else:
                outcome = "draw"              # roughly equal at cap
                result_label = "max_moves_draw"
        else:
            # Stockfish unavailable → treat cap as draw
            outcome      = "draw"
            result_label = "max_moves_draw"

    # Map outcome → per-side reward
    if outcome == "white_wins":
        white_reward, black_reward =  1.0, -1.0
    elif outcome == "black_wins":
        white_reward, black_reward = -1.0,  1.0
    else:
        white_reward, black_reward =  0.0,  0.0

    # Assemble GameSamples
    for (board_tensor, legal_moves, move_index, white_to_move) in raw_samples:
        reward = white_reward if white_to_move else black_reward
        record.samples.append(GameSample(
            board_tensor=board_tensor,
            legal_moves=legal_moves,
            move_index=move_index,
            reward=reward,
        ))

    record.result       = result_label
    record.n_moves      = len(raw_samples)
    record.stockfish_cp = stockfish_cp
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
) -> list[GameRecord]:
    """Run multiple self-play games and return all GameRecords."""
    records = []
    for i in range(n_games):
        record = run_game(
            model,
            max_moves      = max_moves,
            temp_high      = temp_high,
            temp_low       = temp_low,
            temp_threshold = temp_threshold,
            device         = device,
            use_stockfish  = use_stockfish,
        )
        records.append(record)
        if verbose:
            sf_str = (
                f"  sf={record.stockfish_cp:+.0f}cp"
                if record.stockfish_cp is not None else ""
            )
            print(
                f"  Game {i+1:>3}/{n_games} | "
                f"Moves: {record.n_moves:>3} | "
                f"Result: {record.result}{sf_str}"
            )
    return records


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------

def records_to_dataset(records: list[GameRecord]) -> list[GameSample]:
    """Flatten a list of GameRecords into a single list of GameSamples."""
    return [sample for record in records for sample in record]