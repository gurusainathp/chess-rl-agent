"""
policy_network.py
-----------------
CNN-based policy network for the chess RL agent.

Architecture overview
---------------------
The network operates in two stages:

  Stage 1 — Board Encoder  (shared across all moves)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Takes the (13, 8, 8) board tensor and produces a 128-dimensional
  embedding that captures the global position.

      Input : (batch, 13, 8, 8)
      Conv2d(13 → 32,  kernel=3, padding=1) + ReLU
      Conv2d(32 → 64,  kernel=3, padding=1) + ReLU
      Conv2d(64 → 64,  kernel=3, padding=1) + ReLU
      Flatten  →  (batch, 4096)
      Linear(4096 → 256) + ReLU
      Linear(256  → 128)
      Output : (batch, 128)   ← board embedding

  Stage 2 — Move Scoring via Dot Product
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Each legal move is encoded as a 130-dim vector by encode_move(), then
  projected into the same 128-dim space as the board embedding.
  The score for each move is the dot product of its projected vector
  with the board embedding — a single scalar per move.

      move vector  (130,)
           ↓
      Linear(130 → 128)     ← move_projector  (shared weights)
           ↓
      projected move  (128,)
           ↓
      dot( projected_move, board_embedding )  →  scalar logit

  Raw logits are returned — softmax is applied externally (e.g. with
  cross-entropy loss during training, or torch.softmax at inference).

Why dot product?
----------------
The dot product measures alignment between the board embedding and a
move's projected representation.  Moves whose projected vectors point
in a similar direction to the board embedding score highly.  This is
a natural, lightweight scoring mechanism with fewer parameters than a
full MLP head and no need to concatenate embeddings.

Usage
-----
    import chess
    import torch
    from src.environment.board_encoder import encode_board
    from src.models.policy_network import PolicyNetwork

    net   = PolicyNetwork()
    board = chess.Board()

    state  = torch.tensor(encode_board(board)).unsqueeze(0)   # (1, 13, 8, 8)
    moves  = list(board.legal_moves)
    logits = net(state, moves)                                # (1, 20)

    probs  = torch.softmax(logits, dim=1)
    idx    = torch.multinomial(probs, 1).item()
    chosen = moves[idx]
"""

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Move feature encoder
# ---------------------------------------------------------------------------

def encode_move(move: chess.Move) -> torch.Tensor:
    """
    Encode a chess.Move as a 130-dimensional float32 feature vector.

    Layout
    ------
    Indices 0–63   : one-hot over the 64 source squares
    Indices 64–127 : one-hot over the 64 destination squares
    Index 128      : 1.0 if this is a promotion move, else 0.0
    Index 129      : promotion piece type normalised to [0, 1]
                     (queen=5 → 1.0, knight=2 → 0.4) or 0.0

    Parameters
    ----------
    move : chess.Move

    Returns
    -------
    torch.Tensor
        Shape (130,), dtype float32.
    """
    vec = torch.zeros(130, dtype=torch.float32)

    src = chess.square_rank(move.from_square) * 8 + chess.square_file(move.from_square)
    dst = chess.square_rank(move.to_square)   * 8 + chess.square_file(move.to_square)

    vec[src]      = 1.0   # source square one-hot  (indices 0–63)
    vec[64 + dst] = 1.0   # destination one-hot    (indices 64–127)

    if move.promotion is not None:
        vec[128] = 1.0
        vec[129] = move.promotion / 5.0   # queen (5) → 1.0

    return vec


# ---------------------------------------------------------------------------
# PolicyNetwork
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """
    Policy network that scores every legal move in a chess position.

    Scoring mechanism
    -----------------
    Each legal move is encoded, projected to the same 128-dim space as
    the board embedding, and then scored via a dot product:

        logit(move) = board_embedding · project(move_vector)

    This replaces a concatenate-and-MLP approach with a lighter, more
    interpretable bilinear operation.

    Parameters
    ----------
    embedding_dim : int
        Width of the board embedding and move projection space (default 128).
    move_feature_dim : int
        Width of the raw move feature vector from encode_move() (default 130).

    Outputs
    -------
    Raw logits — shape (1, num_legal_moves).
    Apply torch.softmax externally to get a probability distribution.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        move_feature_dim: int = 130,
    ):
        super().__init__()

        self.embedding_dim    = embedding_dim
        self.move_feature_dim = move_feature_dim

        # ------------------------------------------------------------------
        # Stage 1: CNN board encoder
        # Three conv layers preserve spatial dimensions via padding=1.
        # ------------------------------------------------------------------
        self.conv1 = nn.Conv2d(13, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 8 * 8, 256)       # 4096 → 256
        self.fc2 = nn.Linear(256, embedding_dim)      # 256  → 128

        # ------------------------------------------------------------------
        # Stage 2: Move projector
        # Projects a 130-dim move vector into the same 128-dim space as the
        # board embedding so that a dot product gives a meaningful score.
        # ------------------------------------------------------------------
        self.move_projector = nn.Linear(move_feature_dim, embedding_dim, bias=False)
        # bias=False: the board embedding provides all the additive context;
        # a bias here would shift all move scores equally and cancel out.

    # ------------------------------------------------------------------
    # Board encoder — exposed publicly so the value network can reuse it
    # ------------------------------------------------------------------

    def encode_board(self, board_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run the CNN encoder on a board tensor.

        Parameters
        ----------
        board_tensor : torch.Tensor
            Shape (batch, 13, 8, 8), dtype float32.

        Returns
        -------
        torch.Tensor
            Shape (batch, 128) — board position embedding.
        """
        x = F.relu(self.conv1(board_tensor))   # (batch, 32, 8, 8)
        x = F.relu(self.conv2(x))              # (batch, 64, 8, 8)
        x = F.relu(self.conv3(x))              # (batch, 64, 8, 8)
        x = x.flatten(start_dim=1)             # (batch, 4096)
        x = F.relu(self.fc1(x))               # (batch, 256)
        x = self.fc2(x)                        # (batch, 128)
        return x

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_moves: list,
    ) -> torch.Tensor:
        """
        Score all legal moves for a board position.

        Parameters
        ----------
        board_tensor : torch.Tensor
            Shape (1, 13, 8, 8) — single board position.
        legal_moves : list[chess.Move]
            All legal moves available in the current position.

        Returns
        -------
        torch.Tensor
            Shape (1, num_legal_moves) — one raw logit per legal move.

        Raises
        ------
        ValueError
            If legal_moves is empty (game should be over at that point).
        """
        if len(legal_moves) == 0:
            raise ValueError(
                "legal_moves is empty — call is_game_over() before forward()."
            )

        # 1. Encode board → 128-dim embedding  shape: (1, 128)
        board_emb = self.encode_board(board_tensor)

        # 2. Encode and project all moves into the same 128-dim space
        #    move_vecs  : (num_moves, 130)
        #    move_proj  : (num_moves, 128)
        move_vecs = torch.stack([encode_move(m) for m in legal_moves])
        move_proj = self.move_projector(move_vecs)       # (num_moves, 128)

        # 3. Score each move via dot product with the board embedding
        #    board_emb  : (1, 128)  →  transpose to (128, 1)
        #    move_proj @ board_emb.T  →  (num_moves, 1)  →  (1, num_moves)
        logits = (move_proj @ board_emb.T).squeeze(-1).unsqueeze(0)  # (1, num_moves)

        return logits

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def select_move(
        self,
        board_tensor: torch.Tensor,
        legal_moves: list,
        temperature: float = 1.0,
    ) -> chess.Move:
        """
        Sample a move from the softmax policy distribution.

        Parameters
        ----------
        board_tensor : torch.Tensor
            Shape (1, 13, 8, 8).
        legal_moves : list[chess.Move]
        temperature : float
            1.0  → sample proportional to softmax probabilities.
            →0   → behaves identically to greedy_move.
            >1   → more uniform / exploratory distribution.

        Returns
        -------
        chess.Move
        """
        with torch.no_grad():
            logits = self.forward(board_tensor, legal_moves)
            scaled = logits / max(temperature, 1e-8)
            probs  = torch.softmax(scaled, dim=1)
            idx    = torch.multinomial(probs, num_samples=1).item()
        return legal_moves[idx]

    def greedy_move(
        self,
        board_tensor: torch.Tensor,
        legal_moves: list,
    ) -> chess.Move:
        """
        Return the highest-scoring legal move (argmax, no sampling).

        Parameters
        ----------
        board_tensor : torch.Tensor
            Shape (1, 13, 8, 8).
        legal_moves : list[chess.Move]

        Returns
        -------
        chess.Move
        """
        with torch.no_grad():
            logits = self.forward(board_tensor, legal_moves)
            idx    = logits.argmax(dim=1).item()
        return legal_moves[idx]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"PolicyNetwork(\n"
            f"  embedding_dim    = {self.embedding_dim},\n"
            f"  move_feature_dim = {self.move_feature_dim},\n"
            f"  trainable_params = {self.count_parameters():,}\n"
            f")"
        )