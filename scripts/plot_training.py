"""
scripts/plot_training.py
------------------------
Parse training log files and produce training curve plots.

Reads from the timestamped log files written by scripts/train.py
(e.g. logs/train_20260310_230000.log) and produces a PNG with three
subplots:

    1. Training Loss vs Epoch
    2. Winrate vs Random Agent vs Epoch  (only at eval epochs)
    3. Average Game Length vs Epoch

Output
------
    training_curves.png   (or --output path of your choice)

Usage
-----
    # Auto-detect the latest log file
    python scripts/plot_training.py

    # Specify a log file explicitly
    python scripts/plot_training.py --log logs/train_20260310_230000.log

    # Custom output path
    python scripts/plot_training.py --output results/training_curves.png

    # Combine multiple runs on the same plot
    python scripts/plot_training.py \\
        --log logs/train_20260310_230000.log \\
        --log logs/train_20260311_090000.log \\
        --output results/combined.png
"""

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EpochData:
    """All metrics extracted for one training epoch."""
    epoch       : int
    loss        : float | None = None
    avg_length  : float | None = None
    winrate     : float | None = None   # only present at eval epochs
    wins        : int   | None = None
    losses      : int   | None = None


@dataclass
class RunData:
    """All epoch data parsed from one log file."""
    log_path : str
    epochs   : list[EpochData] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Short label for the legend — just the filename without path."""
        return os.path.basename(self.log_path)


# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------

# Regex patterns matching lines written by scripts/train.py
_RE_EPOCH      = re.compile(r"Epoch\s+(\d+)/\d+")
_RE_LOSS       = re.compile(r"Training loss\s*:\s*([\-\d\.]+)")
_RE_AVG_LEN    = re.compile(r"Avg game length\s*:\s*([\d\.]+)")
_RE_WINRATE    = re.compile(r"Winrate\s+([\d\.]+)%")
_RE_EVAL_LINE  = re.compile(
    r"Evaluation\s*:.*W\s+(\d+).*L\s+(\d+).*Winrate\s+([\d\.]+)%"
)


def parse_log(log_path: str) -> RunData:
    """
    Parse a train.py log file and return a RunData object.

    Lines we look for (examples):
        Epoch 5/80
        Training loss     : -0.012341
        Avg game length   : 84.3 half-moves
        Evaluation        : W 28 / L 18 / D 4  →  Winrate 56.0%

    Parameters
    ----------
    log_path : str  Path to the log file.

    Returns
    -------
    RunData
    """
    run    = RunData(log_path=log_path)
    current: EpochData | None = None

    with open(log_path, encoding="utf-8") as f:
        for raw_line in f:
            # Strip log prefix (timestamp + level) if present
            line = re.sub(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[INFO\] ", "", raw_line)
            line = line.strip()

            # New epoch header
            m = _RE_EPOCH.search(line)
            if m:
                if current is not None:
                    run.epochs.append(current)
                current = EpochData(epoch=int(m.group(1)))
                continue

            if current is None:
                continue

            # Loss
            m = _RE_LOSS.search(line)
            if m:
                try:
                    current.loss = float(m.group(1))
                except ValueError:
                    pass
                continue

            # Average game length
            m = _RE_AVG_LEN.search(line)
            if m:
                try:
                    current.avg_length = float(m.group(1))
                except ValueError:
                    pass
                continue

            # Evaluation line — winrate
            m = _RE_EVAL_LINE.search(line)
            if m:
                try:
                    current.wins    = int(m.group(1))
                    current.losses  = int(m.group(2))
                    current.winrate = float(m.group(3)) / 100.0
                except (ValueError, IndexError):
                    pass
                continue

    # Don't forget the last epoch
    if current is not None:
        run.epochs.append(current)

    return run


# ---------------------------------------------------------------------------
# Auto-detect latest log
# ---------------------------------------------------------------------------

def find_latest_log(log_dir: str) -> str | None:
    """
    Find the most recently modified log file in log_dir.

    Returns None if no log files are found.
    """
    pattern = os.path.join(log_dir, "train_*.log")
    files   = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_runs(runs: list[RunData], output_path: str) -> None:
    """
    Produce a three-panel training curve figure and save to output_path.

    Parameters
    ----------
    runs        : list[RunData]   One entry per log file.
    output_path : str             PNG file to write.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — works without a display
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("  ERROR: matplotlib is not installed.")
        print("         Run:  pip install matplotlib")
        sys.exit(1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)
    fig.suptitle("Chess RL Agent — Training Curves", fontsize=14, fontweight="bold")

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, run in enumerate(runs):
        colour = colours[idx % len(colours)]
        label  = run.label

        epochs      = [e.epoch      for e in run.epochs]
        losses      = [e.loss       for e in run.epochs if e.loss      is not None]
        loss_epochs = [e.epoch      for e in run.epochs if e.loss      is not None]
        lengths     = [e.avg_length for e in run.epochs if e.avg_length is not None]
        len_epochs  = [e.epoch      for e in run.epochs if e.avg_length is not None]
        winrates    = [e.winrate    for e in run.epochs if e.winrate   is not None]
        win_epochs  = [e.epoch      for e in run.epochs if e.winrate   is not None]

        # ── Panel 1: Loss ────────────────────────────────────────────
        ax = axes[0]
        if losses:
            ax.plot(loss_epochs, losses, color=colour, label=label,
                    linewidth=1.5, marker="o", markersize=3)
        ax.set_ylabel("Training Loss")
        ax.set_title("Training Loss per Epoch")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.3)
        if len(runs) > 1:
            ax.legend(fontsize=8)

        # Annotate final loss value
        if losses:
            ax.annotate(
                f"{losses[-1]:.4f}",
                xy=(loss_epochs[-1], losses[-1]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, color=colour,
            )

        # ── Panel 2: Winrate ─────────────────────────────────────────
        ax = axes[1]
        if winrates:
            ax.plot(win_epochs, [w * 100 for w in winrates],
                    color=colour, label=label,
                    linewidth=1.5, marker="s", markersize=5)
        ax.axhline(50, color="red", linewidth=1.0, linestyle="--",
                   label="50% baseline" if idx == 0 else None)
        ax.set_ylabel("Winrate vs Random (%)")
        ax.set_title("Winrate vs Random Agent")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax.grid(True, alpha=0.3)
        if win_epochs:
            ax.legend(fontsize=8)

        # Annotate final winrate
        if winrates:
            ax.annotate(
                f"{winrates[-1] * 100:.1f}%",
                xy=(win_epochs[-1], winrates[-1] * 100),
                xytext=(5, -12), textcoords="offset points",
                fontsize=8, color=colour,
            )

        # ── Panel 3: Avg game length ──────────────────────────────────
        ax = axes[2]
        if lengths:
            ax.plot(len_epochs, lengths, color=colour, label=label,
                    linewidth=1.5, marker="^", markersize=3)
        ax.set_ylabel("Avg Game Length (half-moves)")
        ax.set_xlabel("Epoch")
        ax.set_title("Average Game Length per Epoch")
        ax.grid(True, alpha=0.3)
        if len(runs) > 1:
            ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {output_path}")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(runs: list[RunData]) -> None:
    """Print a text summary of parsed metrics to the console."""
    for run in runs:
        print(f"\n  Log : {run.log_path}")
        print(f"  {'─'*50}")

        if not run.epochs:
            print("  No epoch data found in this log file.")
            continue

        print(f"  Epochs parsed : {len(run.epochs)}")

        losses   = [e.loss    for e in run.epochs if e.loss    is not None]
        winrates = [e.winrate for e in run.epochs if e.winrate is not None]
        lengths  = [e.avg_length for e in run.epochs if e.avg_length is not None]

        if losses:
            print(f"  Loss range    : {min(losses):.4f}  →  {max(losses):.4f}")
            print(f"  Final loss    : {losses[-1]:.4f}")
        else:
            print("  Loss          : no data found")

        if winrates:
            print(f"  Winrate range : {min(winrates)*100:.1f}%  →  {max(winrates)*100:.1f}%")
            print(f"  Final winrate : {winrates[-1]*100:.1f}%")
        else:
            print("  Winrate       : no eval data found")

        if lengths:
            print(f"  Avg length    : {sum(lengths)/len(lengths):.1f} half-moves (mean over all epochs)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot training curves from Chess RL log files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--log", type=str, action="append", dest="logs", default=None,
        metavar="PATH",
        help="Path to a train log file.  Repeat to overlay multiple runs. "
             "If omitted, the latest file in --log-dir is used.",
    )
    p.add_argument(
        "--log-dir", type=str, default="logs",
        help="Directory to search for log files when --log is not given.",
    )
    p.add_argument(
        "--output", type=str, default="training_curves.png",
        help="Output path for the PNG figure.",
    )
    p.add_argument(
        "--no-plot", action="store_true",
        help="Print summary to console only — do not produce a PNG.",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # ── Resolve log file(s) ──────────────────────────────────────────
    log_paths = args.logs or []

    if not log_paths:
        latest = find_latest_log(args.log_dir)
        if latest is None:
            print(f"\n  ERROR: No train_*.log files found in '{args.log_dir}'.")
            print(f"         Run training first:  python scripts/train.py")
            sys.exit(1)
        log_paths = [latest]
        print(f"\n  Auto-detected log: {latest}")

    # ── Parse ────────────────────────────────────────────────────────
    runs = []
    for path in log_paths:
        if not os.path.exists(path):
            print(f"  WARNING: log file not found: {path} — skipping.")
            continue
        run = parse_log(path)
        runs.append(run)

    if not runs:
        print("  ERROR: No valid log files to process.")
        sys.exit(1)

    # ── Print summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Chess RL Agent — Training Summary")
    print(f"{'='*60}")
    print_summary(runs)
    print(f"{'='*60}")

    # ── Plot ─────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_runs(runs, args.output)


if __name__ == "__main__":
    main()