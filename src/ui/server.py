"""
src/ui/server.py
----------------
Flask API that bridges the HTML/JS chess UI with the RL model.

Endpoints
---------
  GET  /              Serve index.html
  GET  /status        Model info JSON
  POST /legal         Legal moves for a square
  POST /move          Validate + apply a human move
  POST /ai            Get and apply the model's move

Install dependencies
--------------------
    pip install flask flask-cors

Run
---
    python src/ui/server.py --model models/policy_epoch_0080.pt
    python src/ui/server.py                           # random agent
    python src/ui/server.py --port 5001 --model ...
"""

from __future__ import annotations
import argparse, os, sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import chess, torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from src.environment.board_encoder import encode_board
from src.evaluation.random_agent   import RandomAgent
from src.models.policy_network     import PolicyNetwork
from src.training.train_policy     import load_checkpoint

_HERE = os.path.dirname(os.path.abspath(__file__))

app      = Flask(__name__, static_folder=_HERE)
CORS(app)

_model:     PolicyNetwork | None = None
_epoch:     int | None           = None
_fallback                        = RandomAgent()
_model_path: str | None          = None


def _load(path: str | None) -> None:
    global _model, _epoch, _model_path
    if path and os.path.exists(path):
        _model = PolicyNetwork()
        _epoch = load_checkpoint(path, _model)
        _model.eval()
        _model_path = path
        print(f"  ✓ Model loaded: {path}  (epoch {_epoch})")
    else:
        _model = _epoch = _model_path = None
        print("  ✓ No model — using random agent")


def _get_model_move(board: chess.Board, temperature: float) -> chess.Move:
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError("No legal moves")
    if _model is not None:
        st = torch.tensor(encode_board(board), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return _model.select_move(st, legal, temperature=temperature)
    return _fallback.select_move(board)


def _top_probs(board: chess.Board, temperature: float, n: int = 6) -> list[dict]:
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


def _status_str(board: chess.Board) -> str:
    if board.is_checkmate():           return "checkmate"
    if board.is_stalemate():           return "stalemate"
    if board.is_insufficient_material(): return "insufficient_material"
    if board.is_seventyfive_moves():   return "seventyfive_moves"
    if board.is_fivefold_repetition(): return "fivefold_repetition"
    if board.is_fifty_moves():         return "fifty_moves"
    if board.is_repetition(3):         return "threefold_repetition"
    if board.is_check():               return "check"
    return "ok"


@app.route("/")
def index():
    return send_from_directory(_HERE, "index.html")


@app.route("/status")
def status():
    return jsonify({
        "model_loaded": _model is not None,
        "epoch":        _epoch,
        "model_path":   _model_path,
        "params":       1_157_568 if _model else 0,
        "opponent":     f"ChessRL (epoch {_epoch})" if _model else "Random Agent",
    })


@app.route("/legal", methods=["POST"])
def legal():
    """Return legal destination squares for a given from-square."""
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

    return jsonify({
        "dests": dests,
        "all":   [m.uci() for m in board.legal_moves],
    })


@app.route("/move", methods=["POST"])
def make_move():
    """Apply a human move (UCI) and return the new position."""
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
    status = _status_str(board)

    return jsonify({
        "fen":      board.fen(),
        "san":      san,
        "uci":      uci,
        "status":   status,
        "game_over": board.is_game_over(),
        "result":   board.result() if board.is_game_over() else "*",
        "turn":     "white" if board.turn == chess.WHITE else "black",
    })


@app.route("/ai", methods=["POST"])
def ai_move():
    """Ask the model for its move and return the new position + probabilities."""
    data        = request.get_json(force=True)
    fen         = data.get("fen",         chess.STARTING_FEN)
    temperature = float(data.get("temperature", 0.1))
    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "Invalid FEN"}), 400
    if board.is_game_over():
        return jsonify({"error": "Game over"}), 400

    probs = _top_probs(board, temperature)
    move  = _get_model_move(board, temperature)
    try:    san = board.san(move)
    except: san = move.uci()

    board.push(move)
    status = _status_str(board)

    return jsonify({
        "fen":      board.fen(),
        "san":      san,
        "uci":      move.uci(),
        "status":   status,
        "game_over": board.is_game_over(),
        "result":   board.result() if board.is_game_over() else "*",
        "turn":     "white" if board.turn == chess.WHITE else "black",
        "probs":    probs,
    })


def main():
    p = argparse.ArgumentParser(description="Chess RL — Web UI server")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--port",  type=int, default=5000)
    p.add_argument("--host",  type=str, default="127.0.0.1")
    args = p.parse_args()

    _load(args.model)

    opp = f"ChessRL epoch {_epoch}" if _model else "Random Agent"
    print(f"\n{'='*52}")
    print(f"  Chess RL Agent — Web UI")
    print(f"{'='*52}")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Opponent: {opp}")
    print(f"  Ctrl+C to stop")
    print(f"{'='*52}\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()