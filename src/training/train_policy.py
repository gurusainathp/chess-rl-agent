"""
train_policy.py
---------------
Training loop for the chess policy network.

Each epoch:
  1. Generate self-play games with the current model.
  2. Flatten all game samples into a training dataset.
  3. Compute cross-entropy loss over the chosen moves.
  4. Backpropagate and update model weights with Adam.
  5. Log metrics and optionally save a checkpoint.

Loss formulation
----------------
For each position in the dataset:

    logits = model(board_tensor, legal_moves)   # (1, num_legal_moves)
    loss   = CrossEntropyLoss(logits, target_move_index)

The target is simply the index of the move the model chose during
self-play.  This trains the network to assign higher probability to moves
that were selected in winning games (via the reward-weighted loss described
below) and lower probability to moves from losing games.

Reward-weighted loss
--------------------
Rather than treating all samples equally, each sample's loss contribution
is scaled by its reward:

    weighted_loss = loss * |reward|

This ensures:
  - Wins (+1)  : push strongly toward the chosen move
  - Losses (-1): push strongly AWAY from the chosen move (loss is negated)
  - Draws (0)  : sample contributes nothing to the gradient

Usage
-----
    python scripts/train.py

Or import directly:

    from src.training.train_policy import train, TrainingConfig

    config = TrainingConfig(n_epochs=10, games_per_epoch=20)
    model  = train(config)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn.functional as F

from src.models.policy_network import PolicyNetwork
from src.training.self_play import run_games, records_to_dataset, GameSample
from src.opponents.opponent_pool import OpponentPool


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """
    All training hyperparameters in one place.

    Attributes
    ----------
    n_epochs        : int   Number of training epochs.
    games_per_epoch : int   Self-play games generated per epoch.
    learning_rate   : float Adam learning rate.
    max_moves       : int   Per-game move cap passed to self-play.
    temp_high       : float Early-game sampling temperature.
    temp_low        : float Late-game sampling temperature.
    temp_threshold  : int   Move number at which temp drops to temp_low.
    checkpoint_dir  : str   Directory to save model checkpoints.
    checkpoint_every: int   Save a checkpoint every N epochs (0 = never).
    device          : str   Torch device ('cpu' or 'cuda').
    verbose         : bool  Print per-game self-play output.
    """
    n_epochs         : int   = 50
    games_per_epoch  : int   = 20
    learning_rate    : float = 1e-3
    max_moves        : int   = 200
    temp_high        : float = 1.0
    temp_low         : float = 0.1
    temp_threshold   : int   = 30
    entropy_coeff    : float = 0.01   # λ for entropy regularisation
    checkpoint_dir   : str   = "models"
    checkpoint_every : int   = 10
    log_dir          : str   = "logs"  # directory for timestamped .log files
    use_stockfish    : bool  = True
    eval_games       : int   = 50
    eval_every       : int   = 10     # 0=never, -1=end only
    pgn_dir          : str | None = None
    pgn_every        : int   = 10
    device           : str   = "cpu"
    verbose          : bool  = False
    # ------------------------------------------------------------------
    # Opponent pool
    # ------------------------------------------------------------------
    use_opponent_pool      : bool  = True
    pool_size              : int   = 5
    pool_weight_self       : float = 0.60
    pool_weight_checkpoint : float = 0.20
    pool_weight_random     : float = 0.10
    pool_weight_stockfish  : float = 0.10


# ---------------------------------------------------------------------------
# Epoch metrics container
# ---------------------------------------------------------------------------

@dataclass
class EpochMetrics:
    """Metrics recorded at the end of each epoch."""
    epoch          : int
    loss           : float
    n_samples      : int
    n_games        : int
    wins           : int
    losses         : int
    draws          : int
    max_moves_hit  : int
    duration_sec   : float

    def summary(self) -> str:
        w = self.wins
        l = self.losses
        d = self.draws
        m = self.max_moves_hit
        g = self.n_games
        return (
            f"Epoch {self.epoch:>3} | "
            f"Loss: {self.loss:.4f} | "
            f"Samples: {self.n_samples:>5} | "
            f"W/L/D/Cap: {w}/{l}/{d}/{m} of {g} | "
            f"Time: {self.duration_sec:.1f}s"
        )


# ---------------------------------------------------------------------------
# Single training step over one batch of samples
# ---------------------------------------------------------------------------

def compute_loss(
    model         : PolicyNetwork,
    samples       : list[GameSample],
    device        : str   = "cpu",
    entropy_coeff : float = 0.01,
) -> torch.Tensor:
    """
    Compute reward-weighted REINFORCE loss with entropy regularisation.

    REINFORCE objective: maximise  E[log π(a|s) * R]
    As a minimisation loss:        -log π(a|s) * R  =  CE * R

    Sign convention
    ---------------
    CE loss is always positive.  Multiplying by reward gives:
      Win  (+1): positive → gradient *lowers* CE → raises prob of winning move ✓
      Loss (-1): negative → gradient *raises* CE → lowers prob of losing move  ✓
      Draw ( 0): zero     → no gradient signal                                  ✓

    Critically: wins and losses do NOT cancel — wins minimise (push prob up),
    losses effectively maximise their CE term (push prob down).  Averaging
    over non-zero samples normalises scale without sign cancellation.

    Entropy regularisation
    ----------------------
    -entropy_coeff * H(π)  is subtracted from the loss to encourage
    exploration.  Maximising entropy == minimising its negative.

    Parameters
    ----------
    model         : PolicyNetwork  (must be in train mode)
    samples       : list[GameSample]
    device        : str
    entropy_coeff : float  Weight on entropy bonus (default 0.01)

    Returns
    -------
    torch.Tensor  — scalar loss, ready for .backward()
                    requires_grad=False when all rewards are zero (all draws)
    """
    policy_loss   = torch.zeros((), device=device)
    entropy_sum   = torch.zeros((), device=device)
    n_policy      = 0   # non-zero reward samples
    n_entropy     = 0   # all samples (entropy computed over everything)

    for sample in samples:
        board_t = sample.board_tensor.unsqueeze(0).to(device)   # (1, 13, 8, 8)
        logits  = model(board_t, sample.legal_moves)             # (1, N_legal)
        log_probs = F.log_softmax(logits, dim=-1)                # (1, N_legal)
        probs     = log_probs.exp()

        # Entropy of this position's distribution
        entropy_sum = entropy_sum - (probs * log_probs).sum()
        n_entropy  += 1

        if sample.reward == 0.0:
            continue   # no REINFORCE signal for draws

        target  = torch.tensor([sample.move_index], dtype=torch.long, device=device)
        ce_loss = F.cross_entropy(logits, target)

        # Correct REINFORCE: loss = CE * reward
        #   reward=+1 → minimise CE → reinforce winning move
        #   reward=-1 → negative term → gradient raises CE → suppresses losing move
        policy_loss = policy_loss + ce_loss * sample.reward
        n_policy   += 1

    if n_policy > 0:
        policy_loss = policy_loss / n_policy   # normalise

    if n_entropy > 0:
        entropy_sum = entropy_sum / n_entropy  # mean entropy

    # Total loss: REINFORCE - entropy bonus
    # (subtracting entropy because we *maximise* entropy, i.e. minimise -H)
    total_loss = policy_loss - entropy_coeff * entropy_sum

    return total_loss


# ---------------------------------------------------------------------------
# Logger helper
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str) -> logging.Logger:
    """
    Configure a logger that writes to both console and a timestamped log file.

    Console : plain message, INFO and above
    File    : timestamp + level + message, DEBUG and above

    Returns the 'chess_rl' logger.  Calling setup_logger() again (e.g. on
    resume) clears existing handlers so duplicate lines are never written.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger("chess_rl")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()   # avoid duplicates on re-runs / resume

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file : {log_path}")
    return logger


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model    : PolicyNetwork,
    epoch    : int,
    metrics  : EpochMetrics,
    save_dir : str,
) -> str:
    """
    Save model weights and epoch metadata to disk.

    Parameters
    ----------
    model    : PolicyNetwork
    epoch    : int
    metrics  : EpochMetrics
    save_dir : str  Directory to write into (created if absent).

    Returns
    -------
    str  Path to the saved checkpoint file.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"policy_epoch_{epoch:04d}.pt")

    torch.save(
        {
            "epoch"       : epoch,
            "model_state" : model.state_dict(),
            "metrics"     : {
                "loss"    : metrics.loss,
                "samples" : metrics.n_samples,
                "games"   : metrics.n_games,
            },
        },
        path,
    )
    return path


def load_checkpoint(path: str, model: PolicyNetwork) -> int:
    """
    Load model weights from a checkpoint file.

    Parameters
    ----------
    path  : str              Path to a .pt checkpoint file.
    model : PolicyNetwork    Model instance to load weights into.

    Returns
    -------
    int  The epoch number stored in the checkpoint.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint["epoch"]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    config      : TrainingConfig | None = None,
    model       : PolicyNetwork  | None = None,
    start_epoch : int = 0,
) -> tuple[PolicyNetwork, list[EpochMetrics]]:
    """
    Run the full training loop.

    Parameters
    ----------
    config      : TrainingConfig  Hyperparameters (defaults to TrainingConfig()).
    model       : PolicyNetwork   Model to train (fresh one created if None).
    start_epoch : int             Epoch to resume from (0 = fresh start).
                                  Pass the value returned by load_checkpoint().

    Returns
    -------
    model   : PolicyNetwork      — trained model (weights updated in place).
    history : list[EpochMetrics] — one entry per epoch.
    """
    from src.training.pgn_writer import records_to_pgn, save_pgn
    from src.evaluation.evaluate_model import evaluate, EvaluationConfig

    if config is None:
        config = TrainingConfig()
    if model is None:
        model = PolicyNetwork()

    # ── Logger ────────────────────────────────────────────────────────────
    log = setup_logger(config.log_dir)

    model.to(config.device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: list[EpochMetrics] = []
    train_start = time.time()

    total_epochs = start_epoch + config.n_epochs
    resume_str   = f" (resuming from epoch {start_epoch})" if start_epoch > 0 else ""

    log.info(f"\n{'='*60}")
    log.info(f"  Chess RL — Policy Network Training{resume_str}")
    log.info(f"{'='*60}")
    log.info(f"  Epochs to run   : {config.n_epochs}  (up to epoch {total_epochs})")
    log.info(f"  Games / epoch   : {config.games_per_epoch}")
    log.info(f"  Learning rate   : {config.learning_rate}")
    log.info(f"  Entropy coeff   : {config.entropy_coeff}")
    log.info(f"  Max moves       : {config.max_moves}")
    log.info(f"  Opponent pool   : {'enabled' if config.use_opponent_pool else 'disabled'}")
    log.info(f"  Stockfish       : {'enabled' if config.use_stockfish else 'disabled'}")
    log.info(f"  Checkpoint dir  : {config.checkpoint_dir}")
    log.info(f"  Log dir         : {config.log_dir}")
    log.info(f"  Device          : {config.device}")
    log.info(f"{'='*60}\n")

    # ── Opponent pool ────────────────────────────────────────────────────
    pool: OpponentPool | None = None
    if config.use_opponent_pool:
        pool = OpponentPool(
            checkpoint_dir = config.checkpoint_dir,
            pool_size      = config.pool_size,
            weights        = {
                "self"       : config.pool_weight_self,
                "checkpoint" : config.pool_weight_checkpoint,
                "random"     : config.pool_weight_random,
                "stockfish"  : config.pool_weight_stockfish,
            },
            device         = config.device,
        )
        log.info(f"  {pool.summary()}\n")

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, total_epochs + 1):
        epoch_start = time.time()

        log.info(f"\n{'─'*60}")
        log.info(f"  Epoch {epoch}/{total_epochs}")
        log.info(f"{'─'*60}")

        # 1. Generate games ------------------------------------------------
        model.eval()
        records = run_games(
            model,
            n_games        = config.games_per_epoch,
            max_moves      = config.max_moves,
            temp_high      = config.temp_high,
            temp_low       = config.temp_low,
            temp_threshold = config.temp_threshold,
            device         = config.device,
            use_stockfish  = config.use_stockfish,
            verbose        = config.verbose,
            opponent_pool  = pool,
        )

        # 2. Build flat dataset --------------------------------------------
        samples = records_to_dataset(records)
        if not samples:
            log.info("  No samples generated — skipping epoch.")
            continue

        # 3. Compute loss and update --------------------------------------
        model.train()
        optimiser.zero_grad()
        loss = compute_loss(
            model, samples,
            device        = config.device,
            entropy_coeff = config.entropy_coeff,
        )

        if loss.requires_grad:
            loss.backward()
            optimiser.step()
            log.info(f"  Training loss     : {loss.item():.6f}")
        else:
            log.info("  Training loss     : 0.000000  (all draws — no update)")

        # 4. Record metrics -----------------------------------------------
        wins    = sum(1 for r in records if r.result == "white_wins")
        losses  = sum(1 for r in records if r.result == "black_wins")
        draws   = sum(1 for r in records if r.result == "draw")
        cap_dec = sum(1 for r in records if r.result == "max_moves_reached")
        cap_drw = sum(1 for r in records if r.result == "max_moves_draw")
        sf_used = sum(1 for r in records if r.stockfish_cp is not None)
        avg_len = sum(r.n_moves for r in records) / len(records)

        log.info(f"  Self-play games   : {len(records)}")
        log.info(f"  Samples collected : {len(samples)}")
        log.info(f"  Outcomes          : W {wins} / L {losses} / D {draws} / Cap-decisive {cap_dec} / Cap-draw {cap_drw}")
        log.info(f"  Stockfish evals   : {sf_used}")
        log.info(f"  Avg game length   : {avg_len:.1f} half-moves")

        duration = time.time() - epoch_start
        metrics  = EpochMetrics(
            epoch         = epoch,
            loss          = loss.item(),
            n_samples     = len(samples),
            n_games       = len(records),
            wins          = wins,
            losses        = losses,
            draws         = draws,
            max_moves_hit = cap_dec + cap_drw,
            duration_sec  = duration,
        )
        history.append(metrics)
        log.info(metrics.summary())

        # 5. Checkpoint + pool refresh ------------------------------------
        if config.checkpoint_every > 0 and epoch % config.checkpoint_every == 0:
            path = save_checkpoint(model, epoch, metrics, config.checkpoint_dir)
            log.info(f"  → Checkpoint saved: {path}")
            if pool is not None:
                pool.refresh_checkpoints()
                log.info(f"  → {pool.summary()}")

        # 6. PGN saving ---------------------------------------------------
        if (config.pgn_dir and config.pgn_every > 0
                and epoch % config.pgn_every == 0):
            pgn_text = records_to_pgn(records, epoch=epoch)
            pgn_path = save_pgn(pgn_text, epoch=epoch, pgn_dir=config.pgn_dir)
            log.info(f"  → PGN saved: {pgn_path}")

        # 7. Evaluation vs random agent -----------------------------------
        run_eval = (
            (config.eval_every > 0 and epoch % config.eval_every == 0)
            or (config.eval_every == -1 and epoch == total_epochs)
        )
        if run_eval and config.eval_games > 0:
            eval_cfg = EvaluationConfig(
                n_games     = config.eval_games,
                max_moves   = config.max_moves,
                temperature = config.temp_low,
                device      = config.device,
            )
            result = evaluate(model, eval_cfg)
            log.info(
                f"  Evaluation        : "
                f"W {result.wins} / L {result.losses} / "
                f"D {result.draws + result.max_moves_games}"
                f"  →  Winrate {result.winrate * 100:.1f}%"
            )

        log.info(f"  Epoch time        : {duration:.1f}s")

    total_time = time.time() - train_start
    log.info(f"\n{'='*60}")
    log.info(f"  Training complete")
    log.info(f"  Total time      : {total_time:.1f}s  ({total_time/60:.1f} min)")
    log.info(f"  Epochs trained  : {len(history)}")
    log.info(f"{'='*60}\n")

    return model, history