"""
src/ui/arena_server.py
----------------------
Unified Flask server for the Chess RL Arena.

Three play modes — selected at launch via --mode:

  human-vs-stockfish   You play against Stockfish at any depth.
                       Great for testing yourself or calibrating depth levels.

  model-vs-stockfish   Your trained model faces Stockfish automatically.
                       Streams moves with a configurable delay so you can
                       watch the game unfold.  Outputs live stats.

  human-vs-model       Classic mode — you play against your trained model.
                       (Same as the original server.py)

Endpoints
---------
  GET  /              Serve arena.html
  GET  /status        Session info (mode, opponent names, model epoch, depth)
  POST /legal         Legal destination squares for a given from-square
  POST /move          Validate + apply a human move (human modes only)
  POST /ai            Get and apply the AI/Stockfish move
  POST /autoplay      Play one full game automatically (model-vs-sf mode)
  POST /reset         Reset the board to starting position

Install
-------
    pip install flask flask-cors

Run
---
    # You vs Stockfish depth 5
    python src/ui/arena_server.py --mode human-vs-stockfish --depth 5

    # Your model vs Stockfish depth 3
    python src/ui/arena_server.py --mode model-vs-stockfish \\
        --model models/policy_epoch_0100.pt --depth 3

    # Classic human vs model
    python src/ui/arena_server.py --mode human-vs-model \\
        --model models/policy_epoch_0100.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import chess
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from src.environment.board_encoder import encode_board
from src.evaluation.random_agent   import RandomAgent
from src.models.policy_network     import PolicyNetwork
from src.training.train_policy     import load_checkpoint

_HERE = os.path.dirname(os.path.abspath(__file__))

app  = Flask(__name__, static_folder=_HERE)
CORS(app)

# ---------------------------------------------------------------------------
# Session state (set once at startup, read-only after that)
# ---------------------------------------------------------------------------

_mode         : str                  = "human-vs-model"   # see modes above
_model        : PolicyNetwork | None = None
_model_epoch  : int | None           = None
_model_path   : str | None           = None
_sf_depth     : int                  = 5
_model_temp   : float                = 0.1
_autoplay_delay: float               = 0.8   # seconds between auto-moves
_fallback      = RandomAgent()

# Autoplay state (model-vs-stockfish live streaming)
_autoplay_game : dict = {
    "running"  : False,
    "fen"      : chess.STARTING_FEN,
    "moves"    : [],          # list of {"san", "uci", "color"}
    "result"   : None,        # "1-0" | "0-1" | "1/2-1/2" | None
    "model_color": "white",   # which side the model plays
    "stats"    : {"wins": 0, "losses": 0, "draws": 0, "games": 0},
}
_autoplay_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers — model inference
# ---------------------------------------------------------------------------

def _model_move(board: chess.Board, temperature: float) -> chess.Move:
    """Return the model's chosen move (or random fallback)."""
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError("No legal moves available.")
    if _model is not None:
        st = torch.tensor(encode_board(board), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return _model.select_move(st, legal, temperature=temperature)
    return _fallback.select_move(board)


def _top_probs(board: chess.Board, temperature: float, n: int = 6) -> list[dict]:
    """Return top-N move probabilities from the model."""
    if _model is None:
        return []
    legal = list(board.legal_moves)
    if not legal:
        return []
    st = torch.tensor(encode_board(board), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = _model(st, legal)
        probs  = torch.softmax(logits / max(temperature, 1e-8), dim=1).squeeze(0).tolist()
    pairs = []
    for mv, p in zip(legal, probs):
        try:    san = board.san(mv)
        except: san = mv.uci()
        pairs.append({"san": san, "uci": mv.uci(), "prob": round(p, 4)})
    pairs.sort(key=lambda x: x["prob"], reverse=True)
    return pairs[:n]


# ---------------------------------------------------------------------------
# Helpers — Stockfish
# ---------------------------------------------------------------------------

def _sf_move(board: chess.Board, depth: int) -> chess.Move:
    """Ask Stockfish for the best move at the given depth."""
    from src.opponents.stockfish_agent import StockfishAgent
    with StockfishAgent(depth=depth) as sf:
        return sf.select_move(board)


# ---------------------------------------------------------------------------
# Helpers — board / game utilities
# ---------------------------------------------------------------------------

def _status_str(board: chess.Board) -> str:
    if board.is_checkmate():              return "checkmate"
    if board.is_stalemate():              return "stalemate"
    if board.is_insufficient_material(): return "insufficient_material"
    if board.is_seventyfive_moves():      return "seventyfive_moves"
    if board.is_fivefold_repetition():    return "fivefold_repetition"
    if board.is_fifty_moves():            return "fifty_moves"
    if board.is_repetition(3):            return "threefold_repetition"
    if board.is_check():                  return "check"
    return "ok"


def _board_response(board: chess.Board, move: chess.Move, probs: list | None = None) -> dict:
    """Build a standard response dict after a move has been pushed."""
    try:    san = board.san(move)
    except: san = move.uci()
    # Note: san must be computed BEFORE push; this function receives post-push board.
    # Caller must pass san separately — see call sites below.
    raise RuntimeError("Use _board_response_from_san instead.")


def _build_response(board: chess.Board, san: str, uci: str, probs=None) -> dict:
    status = _status_str(board)
    return {
        "fen"       : board.fen(),
        "san"       : san,
        "uci"       : uci,
        "status"    : status,
        "game_over" : board.is_game_over(),
        "result"    : board.result() if board.is_game_over() else "*",
        "turn"      : "white" if board.turn == chess.WHITE else "black",
        "probs"     : probs or [],
    }


# ---------------------------------------------------------------------------
# Startup loader
# ---------------------------------------------------------------------------

def _load_model(path: str | None) -> None:
    global _model, _model_epoch, _model_path
    if path and os.path.exists(path):
        _model        = PolicyNetwork()
        _model_epoch  = load_checkpoint(path, _model)
        _model.eval()
        _model_path   = path
        print(f"  ✓ Model loaded : {path}  (epoch {_model_epoch})")
    else:
        _model = _model_epoch = _model_path = None
        if path:
            print(f"  ⚠ Model not found at {path} — using random fallback")
        else:
            print("  ✓ No model specified — using random agent")


def _opponent_label() -> str:
    if _mode == "human-vs-stockfish":
        return f"Stockfish depth {_sf_depth}"
    if _mode == "model-vs-stockfish":
        ep = f" (epoch {_model_epoch})" if _model_epoch else ""
        return f"Model{ep} vs Stockfish depth {_sf_depth}"
    # human-vs-model
    ep = f" epoch {_model_epoch}" if _model_epoch else ""
    return f"ChessRL{ep}" if _model else "Random Agent"


# ---------------------------------------------------------------------------
# Routes — shared
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(_HERE, "arena.html")


@app.route("/status")
def status():
    with _autoplay_lock:
        stats = dict(_autoplay_game["stats"])
    return jsonify({
        "mode"         : _mode,
        "opponent"     : _opponent_label(),
        "model_loaded" : _model is not None,
        "model_epoch"  : _model_epoch,
        "model_path"   : _model_path,
        "sf_depth"     : _sf_depth,
        "autoplay_stats": stats,
    })


@app.route("/legal", methods=["POST"])
def legal():
    data   = request.get_json(force=True)
    fen    = data.get("fen", chess.STARTING_FEN)
    square = data.get("square", "")
    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "Invalid FEN"}), 400

    dests = []
    if square:
        try:
            sq    = chess.parse_square(square)
            dests = [chess.square_name(m.to_square)
                     for m in board.legal_moves if m.from_square == sq]
        except ValueError:
            pass

    return jsonify({"dests": dests, "all": [m.uci() for m in board.legal_moves]})


@app.route("/reset", methods=["POST"])
def reset():
    """Reset autoplay state for a fresh game."""
    with _autoplay_lock:
        _autoplay_game["fen"]     = chess.STARTING_FEN
        _autoplay_game["moves"]   = []
        _autoplay_game["result"]  = None
        _autoplay_game["running"] = False
    return jsonify({"fen": chess.STARTING_FEN, "ok": True})


# ---------------------------------------------------------------------------
# Routes — human move (human-vs-stockfish + human-vs-model)
# ---------------------------------------------------------------------------

@app.route("/move", methods=["POST"])
def make_move():
    """Apply a human move and return the updated position."""
    if _mode == "model-vs-stockfish":
        return jsonify({"error": "Human moves disabled in model-vs-stockfish mode"}), 400

    data = request.get_json(force=True)
    fen  = data.get("fen",  chess.STARTING_FEN)
    uci  = data.get("move", "")

    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "Invalid FEN"}), 400
    try:
        move = chess.Move.from_uci(uci)
    except Exception:
        return jsonify({"error": f"Bad UCI: {uci}"}), 400

    if move not in board.legal_moves:
        return jsonify({"error": "Illegal move"}), 400

    try:    san = board.san(move)
    except: san = uci

    board.push(move)
    return jsonify(_build_response(board, san, uci))


# ---------------------------------------------------------------------------
# Routes — AI move
# (human-vs-stockfish: SF plays; human-vs-model: model plays)
# ---------------------------------------------------------------------------

@app.route("/ai", methods=["POST"])
def ai_move():
    """Return the opponent's response move."""
    if _mode == "model-vs-stockfish":
        return jsonify({"error": "Use /autoplay in model-vs-stockfish mode"}), 400

    data        = request.get_json(force=True)
    fen         = data.get("fen",         chess.STARTING_FEN)
    temperature = float(data.get("temperature", _model_temp))

    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "Invalid FEN"}), 400

    if board.is_game_over():
        return jsonify({"error": "Game over"}), 400

    if _mode == "human-vs-stockfish":
        # Stockfish plays
        move = _sf_move(board, _sf_depth)
        probs = []
    else:
        # Model plays (human-vs-model)
        probs = _top_probs(board, temperature)
        move  = _model_move(board, temperature)

    try:    san = board.san(move)
    except: san = move.uci()

    board.push(move)
    return jsonify(_build_response(board, san, move.uci(), probs))


# ---------------------------------------------------------------------------
# Routes — autoplay (model-vs-stockfish streaming)
# ---------------------------------------------------------------------------

@app.route("/autoplay/start", methods=["POST"])
def autoplay_start():
    """
    Begin a new auto-played game between the model and Stockfish.
    The model colour alternates each game (or can be set via model_color param).
    Returns immediately; poll /autoplay/state for updates.
    """
    if _mode != "model-vs-stockfish":
        return jsonify({"error": "Only available in model-vs-stockfish mode"}), 400

    data        = request.get_json(force=True) or {}
    model_color = data.get("model_color", "white")   # "white" | "black"
    temperature = float(data.get("temperature", _model_temp))

    with _autoplay_lock:
        if _autoplay_game["running"]:
            return jsonify({"error": "Game already running"}), 409
        _autoplay_game["fen"]         = chess.STARTING_FEN
        _autoplay_game["moves"]       = []
        _autoplay_game["result"]      = None
        _autoplay_game["running"]     = True
        _autoplay_game["model_color"] = model_color

    def run():
        board        = chess.Board()
        model_is_w   = (model_color == "white")
        move_history = []

        while not board.is_game_over():
            is_model_turn = (board.turn == chess.WHITE) == model_is_w

            if is_model_turn:
                move = _model_move(board, temperature)
            else:
                move = _sf_move(board, _sf_depth)

            try:    san = board.san(move)
            except: san = move.uci()

            color_str = "white" if board.turn == chess.WHITE else "black"
            board.push(move)
            move_history.append({"san": san, "uci": move.uci(), "color": color_str})

            with _autoplay_lock:
                _autoplay_game["fen"]   = board.fen()
                _autoplay_game["moves"] = list(move_history)

            time.sleep(_autoplay_delay)

        result = board.result()
        with _autoplay_lock:
            _autoplay_game["result"]  = result
            _autoplay_game["running"] = False

            # Tally stats from model's perspective
            s = _autoplay_game["stats"]
            s["games"] += 1
            if result == "1/2-1/2":
                s["draws"] += 1
            elif (result == "1-0" and model_is_w) or (result == "0-1" and not model_is_w):
                s["wins"]   += 1
            else:
                s["losses"] += 1

    t = threading.Thread(target=run, daemon=True)
    t.start()

    return jsonify({"ok": True, "model_color": model_color})


@app.route("/autoplay/state", methods=["GET"])
def autoplay_state():
    """Poll for the current board state during an auto-played game."""
    with _autoplay_lock:
        return jsonify({
            "fen"        : _autoplay_game["fen"],
            "moves"      : _autoplay_game["moves"],
            "result"     : _autoplay_game["result"],
            "running"    : _autoplay_game["running"],
            "model_color": _autoplay_game["model_color"],
            "stats"      : dict(_autoplay_game["stats"]),
        })


@app.route("/autoplay/stop", methods=["POST"])
def autoplay_stop():
    """Abort the current auto-played game."""
    with _autoplay_lock:
        _autoplay_game["running"] = False
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global _mode, _sf_depth, _model_temp, _autoplay_delay

    p = argparse.ArgumentParser(
        description="Chess RL Arena — unified play server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["human-vs-stockfish", "model-vs-stockfish", "human-vs-model"],
        default="human-vs-model",
        help=(
            "human-vs-stockfish : you play against Stockfish. "
            "model-vs-stockfish : watch your model face Stockfish. "
            "human-vs-model     : you play against your trained model."
        ),
    )
    p.add_argument("--model",   type=str,   default=None,
                   help="Path to a .pt checkpoint (required for model modes).")
    p.add_argument("--depth",   type=int,   default=5,
                   help="Stockfish search depth (1–20). Lower = weaker & faster.")
    p.add_argument("--temp",    type=float, default=0.1,
                   help="Model sampling temperature (0.01=strongest, 2.0=random).")
    p.add_argument("--delay",   type=float, default=0.8,
                   help="Seconds between moves in model-vs-stockfish autoplay.")
    p.add_argument("--port",    type=int,   default=5000)
    p.add_argument("--host",    type=str,   default="127.0.0.1")
    args = p.parse_args()

    _mode           = args.mode
    _sf_depth       = args.depth
    _model_temp     = args.temp
    _autoplay_delay = args.delay

    # Validate: model modes need either --model or will fall back to random
    if _mode in ("model-vs-stockfish", "human-vs-model"):
        _load_model(args.model)
    else:
        # human-vs-stockfish: no model needed
        print("  ✓ Mode: human-vs-stockfish — no model required")

    # Check Stockfish availability for any mode that uses it
    if _mode in ("human-vs-stockfish", "model-vs-stockfish"):
        from src.opponents.stockfish_agent import stockfish_available
        if not stockfish_available():
            print("\n  ✗ ERROR: Stockfish binary not found.")
            print("    Install Stockfish and set STOCKFISH_PATH or add it to PATH.")
            sys.exit(1)
        print(f"  ✓ Stockfish   : depth {_sf_depth}")

    W = 58
    print(f"\n{'='*W}")
    print(f"  Chess RL Arena")
    print(f"{'='*W}")
    print(f"  Mode          : {_mode}")
    print(f"  URL           : http://{args.host}:{args.port}")
    print(f"  Opponent      : {_opponent_label()}")
    if _mode != "human-vs-stockfish":
        print(f"  Temperature   : {_model_temp}")
    if _mode == "model-vs-stockfish":
        print(f"  Autoplay delay: {_autoplay_delay}s between moves")
    print(f"  Ctrl+C to stop")
    print(f"{'='*W}\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()