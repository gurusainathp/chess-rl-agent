"""
chess_env.py
------------
The core chess environment for the RL agent.

Wraps python-chess to provide a clean gym-style interface:
    env = ChessEnv()
    state = env.reset()
    state, reward, done, info = env.step(move)

Responsibilities:
    1. Initialise the board
    2. Validate moves
    3. Apply moves
    4. Calculate rewards
    5. Detect game end
"""

import chess


# ---------------------------------------------------------------------------
# Reward constants — tweak these to shape agent behaviour
# ---------------------------------------------------------------------------

REWARD_ILLEGAL_MOVE   = -0.5   # Attempted a move not allowed by the rules
REWARD_LEGAL_MOVE     = +0.01  # Any valid move (encourages early exploration)
REWARD_CAPTURE_PAWN   = +0.05  # Captured an opponent's pawn
REWARD_CAPTURE_PIECE  = +0.10  # Captured a more valuable piece (knight and above)
REWARD_LOSE_PIECE     = -0.10  # One of our pieces was captured (detected after move)
REWARD_CHECKMATE_WIN  = +1.00  # We delivered checkmate — the ultimate goal
REWARD_CHECKMATE_LOSS = -1.00  # We were checkmated
REWARD_DRAW           =  0.00  # Stalemate, repetition, insufficient material, etc.

# Map piece types to their capture reward tier
# Pawn → REWARD_CAPTURE_PAWN; everything else → REWARD_CAPTURE_PIECE
_PIECE_CAPTURE_REWARD = {
    chess.PAWN:   REWARD_CAPTURE_PAWN,
    chess.KNIGHT: REWARD_CAPTURE_PIECE,
    chess.BISHOP: REWARD_CAPTURE_PIECE,
    chess.ROOK:   REWARD_CAPTURE_PIECE,
    chess.QUEEN:  REWARD_CAPTURE_PIECE,
}


# ---------------------------------------------------------------------------
# ChessEnv
# ---------------------------------------------------------------------------

class ChessEnv:
    """
    A gym-style chess environment built on top of python-chess.

    Typical usage
    -------------
    env = ChessEnv()
    state = env.reset()

    while not env.is_game_over():
        legal = env.get_legal_moves()
        move  = legal[0]                          # replace with agent policy
        state, reward, done, info = env.step(move)

    Attributes
    ----------
    board : chess.Board
        The underlying python-chess board.  All move logic and legality
        checking is delegated here.
    move_count : int
        Number of half-moves (plies) taken since the last reset.
    """

    def __init__(self):
        self.board: chess.Board = chess.Board()
        self.move_count: int = 0

    # ------------------------------------------------------------------
    # 1. Initialise board
    # ------------------------------------------------------------------

    def reset(self) -> chess.Board:
        """
        Reset the environment to the standard chess starting position.

        Returns
        -------
        chess.Board
            The fresh board state (also accessible via self.board).
        """
        self.board = chess.Board()
        self.move_count = 0
        return self.board

    # ------------------------------------------------------------------
    # 2. Expose state
    # ------------------------------------------------------------------

    def get_state(self) -> chess.Board:
        """
        Return the current board state.

        The board object can be passed directly to board_encoder.py to
        produce the tensor that feeds into the neural network.

        Returns
        -------
        chess.Board
            Current board position.
        """
        return self.board

    def get_legal_moves(self) -> list[chess.Move]:
        """
        Return all legal moves available to the side currently to move.

        Returns
        -------
        list[chess.Move]
            Every move that is legal in the current position.
        """
        return list(self.board.legal_moves)

    # ------------------------------------------------------------------
    # 3 & 4. Apply moves and calculate rewards
    # ------------------------------------------------------------------

    def step(self, move: chess.Move) -> tuple[chess.Board, float, bool, dict]:
        """
        Apply a move and return the resulting (state, reward, done, info).

        If the supplied move is illegal the board is NOT changed and a
        negative reward is returned immediately.

        Parameters
        ----------
        move : chess.Move
            The move to attempt.  Use chess.Move.from_uci("e2e4") or
            pick from get_legal_moves() to guarantee legality.

        Returns
        -------
        state : chess.Board
            Board after the move (unchanged if move was illegal).
        reward : float
            Scalar reward signal for this step.
        done : bool
            True if the game has ended (any reason).
        info : dict
            Diagnostic information — includes 'reason' and 'move_count'.
        """

        # ---- 2. Validate the move ----------------------------------------
        if move not in self.board.legal_moves:
            return self.board, REWARD_ILLEGAL_MOVE, False, {
                "reason": "illegal_move",
                "move": str(move),
                "move_count": self.move_count,
            }

        # ---- Track material before the move so we can detect captures ----
        pieces_before = self._count_pieces()

        # ---- 3. Apply the move -------------------------------------------
        self.board.push(move)
        self.move_count += 1

        # ---- 4. Calculate reward -----------------------------------------
        reward, reward_reason = self._calculate_reward(pieces_before)

        # ---- 5. Detect game end ------------------------------------------
        done = self.is_game_over()

        info = {
            "reason": reward_reason,
            "move": str(move),
            "move_count": self.move_count,
        }

        return self.board, reward, done, info

    # ------------------------------------------------------------------
    # 5. Detect game end
    # ------------------------------------------------------------------

    def is_game_over(self) -> bool:
        """
        Return True if the game has ended for any reason.

        Reasons include checkmate, stalemate, insufficient material,
        the 75-move rule, and fivefold repetition (all handled by
        python-chess automatically).

        Returns
        -------
        bool
        """
        return self.board.is_game_over()

    def get_game_result(self) -> str:
        """
        Return a human-readable description of the game outcome.

        Returns
        -------
        str
            One of: 'white_wins', 'black_wins', 'draw', or 'in_progress'.
        """
        if not self.is_game_over():
            return "in_progress"

        outcome = self.board.outcome()

        if outcome is None:
            return "draw"

        if outcome.winner == chess.WHITE:
            return "white_wins"
        elif outcome.winner == chess.BLACK:
            return "black_wins"
        else:
            return "draw"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _calculate_reward(self, pieces_before: dict) -> tuple[float, str]:
        """
        Compare material before and after a move to determine the reward.

        Called internally by step() after pushing the move onto the board.

        Parameters
        ----------
        pieces_before : dict
            Piece counts from _count_pieces() taken before the move.

        Returns
        -------
        reward : float
        reason : str
            Short label describing what happened (used in info dict).
        """
        # --- Terminal rewards take priority ---

        if self.board.is_checkmate():
            # The side that just moved delivered checkmate
            # board.turn now shows who is to move NEXT (the loser)
            return REWARD_CHECKMATE_WIN, "checkmate_win"

        if self.board.is_stalemate():
            return REWARD_DRAW, "stalemate"

        if self.board.is_insufficient_material():
            return REWARD_DRAW, "insufficient_material"

        if self.board.is_seventyfive_moves():
            return REWARD_DRAW, "75_move_rule"

        if self.board.is_fivefold_repetition():
            return REWARD_DRAW, "fivefold_repetition"

        # --- Capture rewards ---
        pieces_after = self._count_pieces()

        # Who just moved? After push(), board.turn has already flipped.
        # So board.turn == WHITE means Black just moved, and vice-versa.
        opponent_color = self.board.turn          # this side is about to move
        our_color      = not opponent_color        # this side just moved

        reward = REWARD_LEGAL_MOVE
        reason = "legal_move"

        # Check if we captured any opponent pieces
        for piece_type, reward_value in _PIECE_CAPTURE_REWARD.items():
            lost = (pieces_before[opponent_color][piece_type]
                    - pieces_after[opponent_color][piece_type])
            if lost > 0:
                reward += reward_value * lost
                reason  = f"capture_{chess.piece_name(piece_type)}"

        # Check if we lost any of our own pieces (en-passant, etc.)
        for piece_type in _PIECE_CAPTURE_REWARD:
            lost = (pieces_before[our_color][piece_type]
                    - pieces_after[our_color][piece_type])
            if lost > 0:
                reward += REWARD_LOSE_PIECE * lost
                reason  = reason or "lost_piece"

        return reward, reason

    def _count_pieces(self) -> dict:
        """
        Return a nested dict of piece counts by colour and type.

        Returns
        -------
        dict
            {chess.WHITE: {chess.PAWN: int, ...}, chess.BLACK: {...}}
        """
        counts = {
            chess.WHITE: {},
            chess.BLACK: {},
        }
        for color in (chess.WHITE, chess.BLACK):
            for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                               chess.ROOK, chess.QUEEN, chess.KING):
                counts[color][piece_type] = len(
                    self.board.pieces(piece_type, color)
                )
        return counts

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def render(self) -> None:
        """Print the current board to stdout (useful for debugging)."""
        print(self.board)
        print(f"Turn       : {'White' if self.board.turn == chess.WHITE else 'Black'}")
        print(f"Move count : {self.move_count}")
        print(f"Game over  : {self.is_game_over()}")
        print()

    def __repr__(self) -> str:
        return (
            f"ChessEnv("
            f"move_count={self.move_count}, "
            f"turn={'white' if self.board.turn == chess.WHITE else 'black'}, "
            f"game_over={self.is_game_over()})"
        )