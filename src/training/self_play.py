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
  side -1.0; otherwise both sides get draw reward (-1.0).
  If Stockfish is not installed or evaluation fails, the capped game is
  treated as a draw (safe fallback).

Reward schedule: Win=+2.0  Draw=-1.0  Loss=-2.0

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

# Search depth for Stockfish evaluation at move-cap and as an opponent.
# depth=5 plays at solid club level (~1600 ELO) — weak enough that an
# improving model can occasionally beat it, strong enough to give accurate
# positional evaluations at the move cap.
# Do NOT use Limit(time=...) — wall-clock speed varies by machine and can
# accidentally run at full strength on fast hardware.
STOCKFISH_DEPTH: int = 5

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


def _stockfish_evaluate(fen: str, depth: int = STOCKFISH_DEPTH) -> float | None:
    """
    Run Stockfish on the given FEN and return the score from White's POV
    in centipawns, or None if Stockfish is unavailable or fails.

    Uses depth-limited search (not time-limited) for deterministic,
    machine-independent results.

    Parameters
    ----------
    fen   : str  FEN string of the position to evaluate.
    depth : int  UCI search depth (default STOCKFISH_DEPTH = 5).

    Returns
    -------
    float | None
        Centipawn score (positive = White winning, negative = Black winning).
        Mate scores are clamped to ±100_000.
        None if Stockfish is unavailable or evaluation raises an exception.
    """
    sf_path = _SF_PATH
    if sf_path is None:
        return None

    try:
        with chess.engine.SimpleEngine.popen_uci(sf_path) as engine:
            board  = chess.Board(fen)
            result = engine.analyse(
                board,
                chess.engine.Limit(depth=depth),
            )
            score = result["score"].white()   # PovScore from White's POV
            if score.is_mate():
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
    # draw_reason: populated when result starts with "draw" or is a cap-draw
    # values: "" | "stalemate" | "repetition" | "fifty_move" | "insufficient" | "cap_equal" | "checkmate_draw"
    draw_reason   : str              = ""
    n_moves       : int              = 0
    stockfish_cp  : float | None     = None
    opponent_type : str              = "self"   # "self" | "RandomAgent" | "CheckpointAgent"
    model_color   : str              = "white"  # "white" | "black" | "both" (self-play)

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

def _get_draw_reason(board: chess.Board) -> str:
    """
    Return a draw-reason string if the position is an early draw, else "".

    Return values
    -------------
    ""              — not a draw
    "stalemate"     — no legal moves, not in check
    "repetition"    — threefold repetition
    "fifty_move"    — 50-move rule (halfmove clock >= 100)
    "insufficient"  — insufficient mating material
    """
    if board.is_stalemate():
        return "stalemate"
    if board.is_repetition(count=3):
        return "repetition"
    if board.is_fifty_moves():
        return "fifty_move"
    if board.is_insufficient_material():
        return "insufficient"
    return ""


def _is_early_draw(board: chess.Board) -> bool:
    """Return True if _get_draw_reason returns a non-empty string."""
    return _get_draw_reason(board) != ""


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
    opening_moves  : int   = 0,
    opening_prob   : float = 0.5,
) -> GameRecord:
    """
    Play one game and return a GameRecord of training samples.

    Opponent pool support
    ---------------------
    Pass any agent with a ``select_move(board) -> chess.Move`` interface as
    ``opponent``.  When an opponent is supplied only the model's moves are
    stored as GameSamples, and rewards are assigned from the model's POV.
    When ``opponent=None`` (default) the game is pure self-play.

    Opening randomization  (Phase 7)
    ---------------------------------
    When ``opening_moves > 0`` and a random draw beats ``opening_prob``,
    the first ``opening_moves`` half-moves are played randomly by both
    sides before the model takes control.  These random moves are NOT
    stored as training samples — they just set the starting position.

    This prevents opening collapse (the model always playing the same
    first few moves) and exposes it to a much wider variety of positions.

    Recommended values
    ------------------
      opening_moves = 4   (2 moves each side)  ← default when enabled
      opening_prob  = 0.5 (apply to half the games so the model also
                           learns from the standard starting position)

    Parameters
    ----------
    model          : PolicyNetwork
    opponent       : agent | None
    model_color    : bool   chess.WHITE or chess.BLACK.
    max_moves      : int    Half-move cap.
    temp_high      : float  Temperature for early moves.
    temp_low       : float  Temperature for late moves.
    temp_threshold : int    Half-move to switch temperature.
    device         : str    Torch device.
    use_stockfish  : bool   If False, treat move cap as draw.
    opening_moves  : int    Random half-moves to play at game start
                            (0 = disabled, i.e. always start from the
                            standard position).
    opening_prob   : float  Probability [0,1] of applying opening
                            randomization on any given game.

    Returns
    -------
    GameRecord
    """
    is_self_play = (opponent is None)

    model.eval()
    env    = ChessEnv()
    record = GameRecord()
    env.reset()

    # -----------------------------------------------------------------------
    # Opening randomization (Phase 7)
    # Play `opening_moves` random half-moves before the model takes control.
    # These moves are NOT stored as training samples — they just diversify
    # the starting position to prevent opening collapse.
    # -----------------------------------------------------------------------
    import random as _rng
    if opening_moves > 0 and _rng.random() < opening_prob:
        for _ in range(opening_moves):
            if env.is_game_over() or _is_early_draw(env.board):
                break
            legal = env.get_legal_moves()
            if not legal:
                break
            env.step(_rng.choice(legal))

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

    draw_reason = ""

    if env.is_game_over():
        outcome = env.get_game_result()   # 'white_wins' | 'black_wins' | 'draw'
        result_label = outcome
        stockfish_cp = None
        if outcome == "draw":
            draw_reason = "checkmate_draw"   # e.g. fivefold/75-move via python-chess

    else:
        early_draw = _get_draw_reason(env.board)
        if early_draw:
            outcome      = "draw"
            result_label = "draw"
            draw_reason  = early_draw
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
                    outcome      = "draw"
                    result_label = "max_moves_draw"
                    draw_reason  = "cap_equal"
            else:
                outcome      = "draw"
                result_label = "max_moves_draw"
                draw_reason  = "cap_equal"

    # -----------------------------------------------------------------------
    # Map outcome → per-sample reward
    # -----------------------------------------------------------------------

    if is_self_play:
        # Both sides' moves are stored — assign per-side rewards.
        # Reward schedule: Win=+2.0  Draw=0.0  Loss=-2.0
        # Draws give zero signal — REINFORCE requires rewards centred near zero.
        # A draw penalty breaks training because CE loss is unbounded positive,
        # so reward×CE for draws accumulates to -∞ over many epochs.
        if outcome == "white_wins":
            white_reward, black_reward =  2.0, -2.0
        elif outcome == "black_wins":
            white_reward, black_reward = -2.0,  2.0
        else:
            white_reward, black_reward =  0.0,  0.0   # draw = no signal

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

        # Reward schedule: Win=+2.0  Draw=0.0  Loss=-2.0
        if model_won:
            model_reward = 2.0
        elif model_lost:
            model_reward = -2.0
        else:
            model_reward = 0.0   # draw = no signal

        for (board_tensor, legal_moves, move_index, _white_to_move) in raw_samples:
            record.samples.append(GameSample(
                board_tensor=board_tensor,
                legal_moves=legal_moves,
                move_index=move_index,
                reward=model_reward,
            ))

    record.result        = result_label
    record.draw_reason   = draw_reason
    record.n_moves       = len(raw_samples)
    record.stockfish_cp  = stockfish_cp
    record.opponent_type = "self"   # overwritten by run_games when pool is used
    record.model_color   = "both" if is_self_play else ("white" if model_color == chess.WHITE else "black")
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
    opponent_pool  = None,     # OpponentPool | None
    opening_moves  : int   = 0,
    opening_prob   : float = 0.5,
) -> list[GameRecord]:
    """
    Run multiple games and return all GameRecords.

    When opponent_pool is provided each game samples a fresh opponent and
    the model alternates White/Black to remove colour bias.
    When opponent_pool is None all games are pure self-play.

    opening_moves / opening_prob are forwarded to each run_game call
    (see run_game docstring for details).

    Always prints a per-agent breakdown after all games complete.
    """
    from collections import defaultdict

    records: list[GameRecord] = []

    # Per-agent counters: {opp_name: {metric: count}}
    # Metrics: games, win, loss, draw, cap_win, cap_loss, cap_draw
    agent_stats: dict[str, dict] = defaultdict(lambda: dict(
        games=0, win=0, loss=0, draw=0,
        cap_win=0, cap_loss=0, cap_draw=0
    ))

    for i in range(n_games):
        if opponent_pool is not None:
            opponent    = opponent_pool.sample()
            model_color = chess.WHITE if i % 2 == 0 else chess.BLACK
        else:
            opponent    = None
            model_color = chess.WHITE

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
            opening_moves  = opening_moves,
            opening_prob   = opening_prob,
        )

        opp_name             = "self" if opponent is None else type(opponent).__name__
        record.opponent_type = opp_name

        # Tally outcomes from the MODEL's perspective
        # For self-play, model_color alternates so we track raw game result
        s = agent_stats[opp_name]
        s["games"] += 1
        if record.result in ("white_wins", "black_wins"):
            # Decisive game — did the model win?
            is_self_play_game = (opponent is None)
            if is_self_play_game:
                # In self-play the model plays both sides — count once as win
                # (one side wins, the other loses; net = 1 decisive game)
                s["win"] += 1
            else:
                model_won = (
                    (record.result == "white_wins" and model_color == chess.WHITE) or
                    (record.result == "black_wins"  and model_color == chess.BLACK)
                )
                if model_won:
                    s["win"] += 1
                else:
                    s["loss"] += 1
        elif record.result == "draw":
            s["draw"] += 1
        elif record.result == "max_moves_draw":
            s["cap_draw"] += 1
        elif record.result == "max_moves_reached":
            # Stockfish evaluated — determine winner
            cp = record.stockfish_cp
            if cp is not None:
                is_self_play_game = (opponent is None)
                if is_self_play_game:
                    # Self-play: just count as cap_win (decisive result exists)
                    s["cap_win"] += 1
                else:
                    white_better = cp >= STOCKFISH_WIN_THRESHOLD_CP
                    black_better = cp <= -STOCKFISH_WIN_THRESHOLD_CP
                    model_is_white = (model_color == chess.WHITE)
                    model_ahead = (white_better and model_is_white) or                                   (black_better and not model_is_white)
                    if model_ahead:
                        s["cap_win"] += 1
                    else:
                        s["cap_loss"] += 1
            else:
                s["cap_draw"] += 1

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

    # ── Comprehensive per-agent summary ──────────────────────────────────
    total_games = len(records)
    col_w = 7
    agents_sorted = sorted(agent_stats.keys())

    header = (f"  {'Opponent':<20} {'Games':>{col_w}} {'Win':>{col_w}} "
              f"{'Loss':>{col_w}} {'Draw':>{col_w}} "
              f"{'Cap-W':>{col_w}} {'Cap-L':>{col_w}} {'Cap-D':>{col_w}}")
    sep = "  " + "-" * (20 + 7 * (col_w + 1))
    lines = [
        f"  Games this epoch: {total_games}",
        sep, header, sep,
    ]

    tot = dict(games=0, win=0, loss=0, draw=0, cap_win=0, cap_loss=0, cap_draw=0)
    for opp in agents_sorted:
        s = agent_stats[opp]
        # Display name: shorten StockfishAgent(depth=8) → Stockfish(d=8) etc.
        display = opp
        if opp == "self":               display = "Self-play"
        elif "Stockfish" in opp:        display = f"Stockfish(d={s.get('_depth', '?')})"
        elif "Checkpoint" in opp:       display = "Checkpoint"
        elif "Random" in opp:           display = "Random"
        lines.append(
            f"  {display:<20} "
            f"{s['games']:>{col_w}} {s['win']:>{col_w}} {s['loss']:>{col_w}} "
            f"{s['draw']:>{col_w}} {s['cap_win']:>{col_w}} "
            f"{s['cap_loss']:>{col_w}} {s['cap_draw']:>{col_w}}"
        )
        for k in tot:
            tot[k] += s.get(k, 0)

    lines += [
        sep,
        f"  {'TOTAL':<20} "
        f"{tot['games']:>{col_w}} {tot['win']:>{col_w}} {tot['loss']:>{col_w}} "
        f"{tot['draw']:>{col_w}} {tot['cap_win']:>{col_w}} "
        f"{tot['cap_loss']:>{col_w}} {tot['cap_draw']:>{col_w}}",
        sep,
    ]
    print("\n".join(lines))

    return records


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------

def records_to_dataset(records: list[GameRecord]) -> list[GameSample]:
    """Flatten a list of GameRecords into a single list of GameSamples."""
    return [sample for record in records for sample in record]